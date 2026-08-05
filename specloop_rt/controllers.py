"""Controllers for the real system.

These are the same control laws validated against the simulator, re-expressed
against ``StepObservation``/``ControlAction``.  Every controller keeps
continuous internal state and quantizes only at actuation (the low-gain-freeze
bug from the simulator study is fixed here by construction).

The headline comparison for the paper:
  * ``StaticTableSpec``  — reimplements vLLM's shipped open-loop
    ``num_speculative_tokens_per_batch_size`` (k as a step function of batch
    size).  This is the BASELINE.
  * ``ClosedLoopSpec``   — closes the loop on measured acceptance.  This is OURS.

Axis-3 (admission-primary, gamma-secondary): given Axis-1/Axis-2's finding
that the admission cap moves SLO/TTFT far more than gamma ever moves goodput
or acceptance, ``KVAwareAdmit`` + ``GatedSpec`` invert the priority the earlier
controllers implied. ``KVAwareAdmit`` is the primary actuator (TTFT-slack cap,
KV-headroom ceiling); ``GatedSpec`` only closes the acceptance-setpoint loop
in the decode-bound-with-KV-headroom regime Axis-1 located, and pins low
everywhere else so it cannot compete with the admission/KV loop for the same
resource. See scripts/sweep_admit_kv.py for the bracketing-arm sweep.
"""
from __future__ import annotations

import math
from bisect import bisect_right
from typing import Dict, List, Optional, Tuple

from .interface import ControlAction, Controller, StepObservation


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ==========================================================================
# Speculation controllers (actuate gamma)
# ==========================================================================


class StaticSpec(Controller):
    """Fixed k. k=0 disables speculation. The trivial baseline."""
    name = "static-spec"

    def __init__(self, gamma: int = 4):
        self.gamma = gamma

    def on_step(self, obs):
        return ControlAction(gamma=self.gamma)


class StaticTableSpec(Controller):
    """vLLM's shipped open-loop policy: k = f(batch_size) via a step table.

    Faithful reimplementation of ``num_speculative_tokens_per_batch_size``
    (list of [start_bs, end_bs, k]).  This is the production baseline the paper
    argues against: it couples k to concurrency WITHOUT sensing acceptance, so
    it cannot react when a same-size batch shifts to low-acceptance traffic.
    """
    name = "static-table-spec"

    def __init__(self, table: Optional[List[Tuple[int, int, int]]] = None):
        # default mirrors the vLLM docs example for eagle3
        self.table = table or [(1, 16, 5), (17, 32, 4), (33, 64, 3),
                               (65, 128, 1), (129, 512, 0)]
        self._starts = [t[0] for t in self.table]

    def _lookup(self, bs: int) -> int:
        for lo, hi, k in self.table:
            if lo <= bs <= hi:
                return k
        return self.table[-1][2]

    def on_step(self, obs):
        return ControlAction(gamma=self._lookup(obs.num_running))


class ClosedLoopSpec(Controller):
    """OURS: setpoint controller closing the loop on measured acceptance.

    Setpoint: the largest gamma whose predicted marginal acceptance
    alpha^gamma still clears ``threshold``.  Proportional approach with a
    deadband so it settles in isolation.  ``period`` decouples the control rate
    from the step rate.
    """
    name = "closed-loop-spec"

    def __init__(self, threshold: float = 0.45, gain: float = 0.5,
                 deadband: float = 0.5, period: int = 4,
                 gamma_min: int = 0, gamma_max: int = 8, gamma_init: int = 4):
        self.threshold = threshold
        self.gain = gain
        self.deadband = deadband
        self.period = max(1, period)
        self.gamma_min, self.gamma_max = gamma_min, gamma_max
        self._g = float(gamma_init)
        self.gamma = gamma_init

    def reset(self):
        self._g = float(self.gamma)

    def on_step(self, obs):
        if obs.step % self.period != 0:
            return ControlAction(gamma=self.gamma)
        a = _clamp(obs.accept_rate_ema, 1e-3, 0.999)
        target = math.log(self.threshold) / math.log(a)
        err = target - self._g
        if abs(err) > self.deadband:
            self._g = _clamp(self._g + self.gain * err, self.gamma_min, self.gamma_max)
        self.gamma = int(round(self._g))
        return ControlAction(gamma=self.gamma)


class DSDESpec(Controller):
    """Per-request ragged k from each request's own acceptance EMA.

    Mirrors the vLLM DynamicProposer / DSDE control law.  Here it emits a
    per_request_gamma map; the patch layer applies it via the proposer hook.
    (In this harness the per-request acceptance is maintained patch-side and
    surfaced through obs; see vllm_patch for the wiring.)
    """
    name = "dsde-spec"

    def __init__(self, target: float = 0.5, gain: float = 0.5, period: int = 4,
                 gamma_min: int = 1, gamma_max: int = 8):
        self.target = target
        self.gain = gain
        self.period = max(1, period)
        self.gamma_min, self.gamma_max = gamma_min, gamma_max
        self._per: Dict[str, float] = {}

    def on_step(self, obs):
        # The global k is advisory; ragged map is applied per request by the
        # proposer patch. We still return a batch-mean gamma for logging.
        return ControlAction(gamma=obs.gamma_current)


class GatedSpec(Controller):
    """Secondary loop: gamma is a regime-gated trim, not a primary knob.

    Axis-1 found decode_bound/both_bound only at the top of the rate ramp;
    Axis-2 found gamma's effect on goodput/acceptance stays under ~10% of the
    mean at EVERY rtype/rate/cap combination tried. Given that, closing a
    proportional loop on gamma everywhere just adds a second actuator that can
    fight the admission/KV loop for no measured benefit, and risks reproducing
    the simulator's batch-aggregate-acceptance limit cycle (see
    specloop/README.md "Status of the findings") on the one signal (measured
    acceptance) both loops can plausibly share.

    So: pin low (``gamma_floor``) unless BOTH hold this step:
      * decode-bound, i.e. tpot_ema has used up ``decode_bound_util`` of its
        SLO (the regime Axis-1 says gamma can matter in) -- gated on
        long-window state, not a single-step read, else load-testing near the
        threshold would toggle the mode every period like a bang-bang law
        (the exact simulator bug this codebase's README documents fixing).
      * KV has headroom (``kv_headroom_frac``) -- speculation reserves gamma
        KV slots per request before acceptance is known (see
        specloop/README.md cost-model note); adapting gamma upward under KV
        pressure would compete with the admission/KV loop for the same
        resource instead of trimming a genuine decode-bound slack.
    When gated in, delegates to the same acceptance-setpoint law as
    ClosedLoopSpec so the "gamma tracks measured acceptance, not a workload
    label" behavior is identical -- only the gate is new.
    """
    name = "gated-spec"

    def __init__(self, threshold: float = 0.15, gain: float = 0.5,
                 deadband: float = 0.5, period: int = 4,
                 gamma_min: int = 0, gamma_max: int = 8, gamma_init: int = 4,
                 gamma_floor: int = 1, decode_bound_util: float = 0.85,
                 kv_headroom_frac: float = 0.15):
        # threshold default: the Axis-3 sweep measured mean_accept_rate in
        # [0.35, 0.46] across every rtype/rate cell on Qwen2.5-7B + ngram
        # speculation (results_gpu_sweep/axis3/grid.json). The original
        # ClosedLoopSpec default of 0.45 solves target=log(threshold)/log(a)
        # to ~1.0 at those acceptance rates, so the gate opened correctly but
        # the setpoint law always answered "gamma=1" -- indistinguishable
        # from the floor, so the gated arm never departed from StaticSpec(1)
        # in that sweep despite the gate mechanism itself working (verified
        # directly against a synthetic decode-bound/KV-headroom observation).
        # 0.15 targets gamma in the 2-2.4 range at the same measured
        # acceptance rates -- a real, non-trivial speculation depth to test
        # against, not a recalibration proven optimal; re-derive from
        # mean_accept_rate on whatever model/draft-method combination this
        # runs against next, since the right threshold is a property of that
        # combination's real acceptance behavior, not a universal constant.
        self.threshold = threshold
        self.gain = gain
        self.deadband = deadband
        self.period = max(1, period)
        self.gamma_min, self.gamma_max = gamma_min, gamma_max
        self.gamma_floor = gamma_floor
        self.decode_bound_util = decode_bound_util
        self.kv_headroom_frac = kv_headroom_frac
        self._g = float(gamma_init)
        self.gamma = gamma_init
        self._gated_in = False

    def reset(self):
        self._g = float(self.gamma)
        self._gated_in = False

    def _gate_open(self, obs) -> bool:
        decode_bound = obs.tpot_ema >= self.decode_bound_util * max(obs.tpot_slo_s, 1e-9)
        kv_headroom = (1.0 - obs.kv_used_frac) >= self.kv_headroom_frac
        return decode_bound and kv_headroom

    def on_step(self, obs):
        if obs.step % self.period != 0:
            return ControlAction(gamma=self.gamma)
        self._gated_in = self._gate_open(obs)
        if not self._gated_in:
            self._g = float(self.gamma_floor)
            self.gamma = self.gamma_floor
            return ControlAction(gamma=self.gamma)
        a = _clamp(obs.accept_rate_ema, 1e-3, 0.999)
        target = math.log(self.threshold) / math.log(a)
        err = target - self._g
        if abs(err) > self.deadband:
            self._g = _clamp(self._g + self.gain * err, self.gamma_min, self.gamma_max)
        self.gamma = int(round(self._g))
        return ControlAction(gamma=self.gamma)


class HillClimbSpec(Controller):
    """Batch-gated hill-climb on measured ITL. Monitors obs.tpot_ema (ground
    truth ITL) and obs.num_running (live batch B), and searches for the
    ITL-minimizing k directly rather than reading a pre-fit curve or tracking
    acceptance.

    Why not acceptance (ClosedLoopSpec/GatedSpec) or a static k*(B) lookup:
    the Mixtral-8x7B-FP8/ngram roofline sweep (results_gpu_sweep/
    axis4_roofline_moe_fit) found ITL(k) has an INTERIOR optimum at every
    batch tested (B=8..32), with acceptance falling monotonically across the
    whole k range -- so acceptance cannot localize the optimum, it only says
    "less accepted per token as k grows," true on both sides of the peak.
    The optimal k also shifts with B (roughly 4/6/3/2 across B=8/16/24/32),
    so any single fixed k is wrong somewhere in that range. This controller
    hill-climbs the metric that actually has the interior optimum (ITL
    itself) instead of a proxy that doesn't.

    Why batch-gated, and why a fitted formula rather than the raw peaks: an
    unconstrained hill-climb wastes early steps probing k values the same
    sweep already shows are wrong for the current regime. The first version
    of this window used the raw per-B row-minimum (4/6/3/2 at B=8/16/24/32)
    -- but those are single-seed point estimates, noisy, and don't move
    smoothly with B (B=16's "6" breaks monotonicity because its optimum
    region is genuinely broad/flat, not because k=6 is meaningfully better
    than k=4 or k=5 there -- all three are within 3% of the row minimum).
    Fitting ITL(B,k) = c0 + c1*B + c2*k + c3*k^2 + c4*B*k by least squares
    across all 28 non-baseline cells (not just the 4 row-minima) gives R^2 =
    0.94 and, from dITL/dk = 0, a closed form:

        k*(B) = -(c2 + c4*B) / (2*c3)  ~=  6.66 - 0.095*B

    This borrows statistical strength from every cell instead of just the
    single noisiest point per row, and gives a smooth, monotonic k*(B) the
    raw peaks don't cleanly show. ``window_halfwidth`` (default 2) bounds the
    live search to k*(B) +- that margin (clamped to [1, 8]) -- a window
    around the formula's prediction, not the formula's prediction itself.
    The hill-climb still does the work of finding the actual optimum inside
    that window; the formula only narrows where it looks, so a live run
    still self-corrects if real conditions (workload, rate, hardware) diverge
    from what the fit sweep measured. Re-fit ``_KSTAR_INTERCEPT``/``_KSTAR_SLOPE``
    against a new B x k sweep before trusting this on a different model/
    hardware pairing -- these coefficients are specific to Mixtral-8x7B-FP8
    + ngram on one A100, not a hardware law.

    Algorithm: settle-then-compare, NOT compare-every-period. ``obs.tpot_ema``
    is itself an EMA (alpha=0.1 per scheduler step, see vllm_patch/
    scheduler_patch.py's ``_sl_ema["tpot"]``) with a ~10-step time constant --
    an early version of this controller compared it every ``period=4`` steps
    and never converged: a smoke test at B=16 showed gamma still bouncing
    across the ENTIRE search window after 4400+ saturated steps, because at 4
    steps the EMA has only moved ~34% of the way toward reflecting the new
    k's true effect (1-(1-0.1)^4). Each comparison was reading transient, not
    settled, signal. Fixed by holding k fixed for ``settle_steps`` (default
    40, ~4x the EMA time constant, empirically >95% settled) before every
    comparison, and comparing against the ITL measured at the END of the
    PREVIOUS settle window rather than the previous single step.

    Every ``settle_steps`` steps: clamp gamma into the current batch-derived
    window, read ``obs.tpot_ema`` (now settled), and compare to the reading
    from the last window. If it improved, keep stepping in the same
    direction; if it got worse by more than ``deadband_frac`` (relative, so
    the threshold scales with the operating point), reverse direction.
    Bounces off window edges rather than getting stuck there.
    """
    name = "hillclimb-spec"

    # Least-squares fit of ITL(B,k) = c0 + c1*B + c2*k + c3*k^2 + c4*B*k
    # across all 28 non-baseline cells of axis4_roofline_moe_fit/grid.json
    # (R^2 = 0.94). k*(B) = -(c2 + c4*B) / (2*c3) reduces to this linear form.
    # Mixtral-8x7B-FP8 + ngram, 1x A100-80GB, code/HumanEval, rate=8 -- refit
    # for a different model/hardware/workload before trusting this elsewhere.
    _KSTAR_INTERCEPT = 6.66
    _KSTAR_SLOPE = -0.095
    _K_HARD_MIN, _K_HARD_MAX = 1, 8

    @classmethod
    def _kstar(cls, num_running: int) -> float:
        return cls._KSTAR_INTERCEPT + cls._KSTAR_SLOPE * num_running

    def __init__(self, settle_steps: int = 40, deadband_frac: float = 0.03,
                 gamma_init: int = 4, window_halfwidth: int = 2,
                 avg_last_n: Optional[int] = None):
        self.settle_steps = max(1, settle_steps)
        self.deadband_frac = deadband_frac
        self.window_halfwidth = max(0, window_halfwidth)
        # Comparisons average the last avg_last_n readings collected during
        # each settle window, not a single point-sample at the window's end.
        # Default: the whole window. WHY: found via offline replay against
        # real B=32 tpot_ema traces (results_gpu_sweep/
        # axis5_hillclimb_vs_static_live) after a live GPU run showed this
        # controller does not beat the best static k -- comparing single
        # readings is comparing noise (measured tpot_ema stdev ~7.7% of the
        # mean at B=32, i.e. LARGER than deadband_frac's default 3%), so a
        # single sample cannot reliably distinguish adjacent k's whose true
        # ITL means differ by a similar few percent. Averaging avg_last_n
        # samples cuts comparison noise by ~sqrt(n); replay showed n=40
        # (the full settle window) reduces steady-state ITL by ~2.6% vs
        # n=1 and collapses the steady-state gamma histogram from a 5-wide
        # spread (2-6) down to 2 adjacent values, without changing
        # settle_steps or adding wall-clock cost (the samples are free --
        # every step in the settle window already computes obs.tpot_ema).
        self.avg_last_n = avg_last_n if avg_last_n is not None else self.settle_steps
        self.gamma = gamma_init
        self._direction = 1          # +1 climbing up, -1 climbing down
        self._last_itl: Optional[float] = None
        self._window_start_step: Optional[int] = None
        self._readings: List[float] = []

    def reset(self):
        self._direction = 1
        self._last_itl = None
        self._window_start_step = None
        self._readings = []

    def _window(self, num_running: int) -> Tuple[int, int]:
        center = round(self._kstar(num_running))
        k_min = _clamp(center - self.window_halfwidth, self._K_HARD_MIN, self._K_HARD_MAX)
        k_max = _clamp(center + self.window_halfwidth, self._K_HARD_MIN, self._K_HARD_MAX)
        return int(k_min), int(k_max)

    def on_step(self, obs):
        if self._window_start_step is None:
            self._window_start_step = obs.step
        elapsed = obs.step - self._window_start_step
        # collect readings for the trailing avg_last_n steps of this window
        if elapsed >= self.settle_steps - self.avg_last_n:
            self._readings.append(obs.tpot_ema)
        if elapsed < self.settle_steps:
            return ControlAction(gamma=self.gamma)
        self._window_start_step = obs.step

        k_min, k_max = self._window(obs.num_running)
        self.gamma = _clamp(self.gamma, k_min, k_max)

        itl = sum(self._readings) / len(self._readings) if self._readings else obs.tpot_ema
        self._readings = []
        # Hold position when the windowed comparison is ambiguous (within
        # the deadband either direction) instead of always taking another
        # step in the current direction -- without this, a controller
        # sitting exactly at the optimum still takes a random-walk step
        # every window purely from residual averaging noise, which is what
        # produced the persistent 2-3-value oscillation this fix targets.
        move = True
        if self._last_itl is not None and itl > 0:
            # Relative deadband: at higher ITL operating points a fixed
            # absolute threshold is too tight (noise dominates); at low ITL
            # it's too loose (real signal gets ignored as noise).
            rel_change = (itl - self._last_itl) / self._last_itl
            if rel_change > self.deadband_frac:
                self._direction *= -1
            elif abs(rel_change) <= self.deadband_frac:
                move = False
        self._last_itl = itl

        if move:
            proposed = self.gamma + self._direction
            if proposed > k_max:
                self._direction = -1
                proposed = self.gamma + self._direction
            elif proposed < k_min:
                self._direction = 1
                proposed = self.gamma + self._direction
            self.gamma = _clamp(proposed, k_min, k_max)
        return ControlAction(gamma=self.gamma)


SPEC_CONTROLLERS = {
    "static": StaticSpec,
    "static-table": StaticTableSpec,
    "closed-loop": ClosedLoopSpec,
    "dsde": DSDESpec,
    "gated": GatedSpec,
    "hillclimb": HillClimbSpec,
}


# ==========================================================================
# Admission controllers (actuate max_num_seqs)
# ==========================================================================


class StaticAdmit(Controller):
    name = "static-admit"

    def __init__(self, max_num_seqs: int = 64):
        self.max_num_seqs = max_num_seqs

    def on_step(self, obs):
        return ControlAction(max_num_seqs=self.max_num_seqs)


class SlackAdmit(Controller):
    """Slack-guided admission on the TPOT SLO. Deadband for isolation stability."""
    name = "slack-admit"

    def __init__(self, target_util: float = 0.85, gain: float = 0.5,
                 deadband: float = 0.05, period: int = 4,
                 batch_min: int = 1, batch_max: int = 256, init: int = 64):
        self.target_util = target_util
        self.gain = gain
        self.deadband = deadband
        self.period = max(1, period)
        self.batch_min, self.batch_max = batch_min, batch_max
        self._b = float(init)
        self.max_num_seqs = init

    def on_step(self, obs):
        if obs.step % self.period != 0:
            return ControlAction(max_num_seqs=self.max_num_seqs)
        tpot = obs.tpot_ema
        slack = (obs.tpot_slo_s * self.target_util - tpot) / max(obs.tpot_slo_s, 1e-9)
        if abs(slack) > self.deadband:
            delta = self.gain * 8.0 * slack
            if obs.kv_used_frac > 0.92:
                delta = min(delta, -self.gain * 4.0)
            self._b = _clamp(self._b + delta, self.batch_min, self.batch_max)
        self.max_num_seqs = int(round(self._b))
        return ControlAction(max_num_seqs=self.max_num_seqs)


class _TrendAwareQueueTerm:
    """Shared queue-slack-plus-trend term for TTFTSlackAdmit and KVAwareAdmit.

    The level-only version (waiting_ratio vs. target_util) treats a queue that
    is large-but-draining the same as one that is large-and-growing -- it
    waits for the ratio itself to fall before loosening the cap, which is
    slower than it needs to be whenever the queue is already recovering. This
    adds a trend term: an EMA of the step-over-step change in waiting_ratio.
    A negative trend (queue shrinking) adds extra slack on top of the level
    term, so the cap loosens sooner during a genuine recovery; a positive
    trend (queue still growing) subtracts, tightening faster than the level
    alone would once growth is detected rather than waiting for the level to
    cross target_util. trend_gain=0 recovers the original level-only law
    exactly (trend term is inert), so this is a strict extension, not a
    replacement.
    """

    def __init__(self, target_util: float, trend_gain: float, trend_ema_beta: float):
        self.target_util = target_util
        self.trend_gain = trend_gain
        self.trend_ema_beta = trend_ema_beta
        self._prev_ratio = None
        self._trend_ema = 0.0

    def slack(self, obs, cap_for_ratio: float) -> float:
        # waiting_ratio is unbounded (num_waiting can be many multiples of the
        # cap under real queue collapse), so slack is left unclamped -- an
        # earlier version clamped waiting_ratio to [0,4] before taking the
        # slack, which saturated the signal at exactly the severe-backup case
        # this term exists to catch (waiting=200 at cap=64 produced only a
        # barely-past-deadband nudge instead of a sharp contraction; caught by
        # a sanity check in this module's own test scenarios, not a GPU run).
        waiting_ratio = obs.num_waiting / max(1, cap_for_ratio)
        if self._prev_ratio is not None:
            delta = waiting_ratio - self._prev_ratio
            self._trend_ema = (1 - self.trend_ema_beta) * self._trend_ema + self.trend_ema_beta * delta
        self._prev_ratio = waiting_ratio
        level_slack = self.target_util - waiting_ratio
        # trend_ema > 0 (queue growing) subtracts from slack (tighten sooner);
        # trend_ema < 0 (queue draining) adds to slack (loosen sooner).
        return level_slack - self.trend_gain * self._trend_ema


class _PredictedWaitTerm:
    """Queue term that senses predicted TTFT directly, not a queue-to-cap ratio.

    ``_TrendAwareQueueTerm`` compares ``num_waiting/cap`` against a target
    ratio. On the Llama-3.1-8B/A5000 sweeps that sensor was numb exactly where
    it mattered: at rate=2, TTFT p99 was 37s against a 2.0s SLO (breached 19x)
    while num_waiting peaked at 37 against cap=256 -- a ratio of 0.14, far
    under target_util=0.7, so the cap never left 256 for the entire run
    (verified in adaptive-cap-only_reason_r2.0_s0/steps.jsonl). The loop was
    not losing to the static baseline; it never actuated at all.

    The failure is that queue-to-cap ratio is a proxy for waiting time, and a
    bad one when the cap is generous: a queue can be short relative to a large
    cap while each request in it still waits far past its TTFT SLO, because
    what sets waiting time is queue length divided by *service rate*, not
    divided by the cap.

    So sense the quantity the SLO is written against. By Little's Law the wait
    a newly-arriving request faces is approximately

        predicted_wait = num_waiting / max(finish_rate, eps)

    where ``finish_rate`` is requests completing per second, estimated as
    ``num_running / mean_request_duration``. We do not have per-request
    duration in StepObservation, so use the decode-side identity: a request
    running for ``L`` output tokens at ``tpot_ema`` seconds/token occupies a
    slot for ``L * tpot_ema`` seconds, giving

        finish_rate ~= num_running / (L_est * tpot_ema)

    ``L_est`` (``est_output_len``) is the one free parameter and is a workload
    property -- set it from the corpus's mean output length. Slack is then the
    normalized headroom against the TTFT SLO, so the law tightens as soon as
    predicted wait eats into the SLO budget rather than waiting for a ratio
    that may never move.
    """

    def __init__(self, ttft_target_util: float, est_output_len: float,
                 trend_gain: float, trend_ema_beta: float):
        self.ttft_target_util = ttft_target_util
        self.est_output_len = est_output_len
        self.trend_gain = trend_gain
        self.trend_ema_beta = trend_ema_beta
        self._prev_wait = None
        self._trend_ema = 0.0

    def predicted_wait(self, obs) -> float:
        # Slot service time: how long one running request holds its slot.
        service_s = max(self.est_output_len * max(obs.tpot_ema, 1e-6), 1e-6)
        finish_rate = max(obs.num_running, 1) / service_s
        return obs.num_waiting / max(finish_rate, 1e-9)

    def slack(self, obs) -> float:
        wait = self.predicted_wait(obs)
        if self._prev_wait is not None:
            delta = wait - self._prev_wait
            self._trend_ema = ((1 - self.trend_ema_beta) * self._trend_ema
                               + self.trend_ema_beta * delta)
        self._prev_wait = wait
        budget = self.ttft_target_util * max(obs.ttft_slo_s, 1e-9)
        # Normalize by the budget so gain is dimensionless and comparable to
        # the ratio-based law's gain; positive slack = wait is under budget.
        level_slack = (budget - wait) / budget
        # trend > 0 (wait growing) subtracts slack -> tighten sooner.
        return level_slack - self.trend_gain * (self._trend_ema / budget)


class TTFTPredictiveAdmit(Controller):
    """Admission driven by *predicted* TTFT against the TTFT SLO.

    Same actuation shape as ``TTFTSlackAdmit`` (proportional, periodic,
    deadbanded, clamped) but with ``_PredictedWaitTerm`` as the sensor instead
    of ``_TrendAwareQueueTerm``. This is the "rewire the sensor to the SLO
    quantity" fix; everything else about the loop is held constant so the
    comparison against ``ttft-slack`` isolates the sensor change alone.
    """
    name = "ttft-predictive-admit"

    def __init__(self, ttft_target_util: float = 0.7, est_output_len: float = 840.0,
                 gain: float = 0.5, deadband: float = 0.05, period: int = 4,
                 batch_min: int = 1, batch_max: int = 256, init: int = 64,
                 trend_gain: float = 0.5, trend_ema_beta: float = 0.3):
        self.gain = gain
        self.deadband = deadband
        self.period = max(1, period)
        self.batch_min, self.batch_max = batch_min, batch_max
        self._b = float(init)
        self.max_num_seqs = init
        self._queue_term = _PredictedWaitTerm(ttft_target_util, est_output_len,
                                              trend_gain, trend_ema_beta)

    def reset(self):
        pass

    def on_step(self, obs):
        if obs.step % self.period != 0:
            return ControlAction(max_num_seqs=self.max_num_seqs)
        slack = self._queue_term.slack(obs)
        if abs(slack) > self.deadband:
            self._b = _clamp(self._b + self.gain * 8.0 * slack,
                             self.batch_min, self.batch_max)
        self.max_num_seqs = int(round(self._b))
        return ControlAction(max_num_seqs=self.max_num_seqs)


class KVPredictiveAdmit(TTFTPredictiveAdmit):
    """``TTFTPredictiveAdmit`` + the same proportional KV ceiling as
    ``KVAwareAdmit``, combined as ``min``.

    Keeps the admission-primary/KV-safety structure while swapping in the
    working sensor, so the KV ceiling's incremental effect can be measured on
    top of a cap loop that actually actuates.
    """
    name = "kv-predictive-admit"

    def __init__(self, kv_target: float = 0.80, kv_min_frac: float = 0.30, **kw):
        super().__init__(**kw)
        self.kv_target = kv_target
        self.kv_min_frac = kv_min_frac

    def _kv_ceiling(self, obs) -> float:
        kv = obs.kv_used_frac
        if kv <= self.kv_target:
            return self.batch_max
        over = (kv - self.kv_target) / max(1e-9, 1.0 - self.kv_target)
        floor = self.kv_min_frac * self.batch_max
        return self.batch_max - over * (self.batch_max - floor)

    def on_step(self, obs):
        if obs.step % self.period != 0:
            return ControlAction(max_num_seqs=self.max_num_seqs)
        slack = self._queue_term.slack(obs)
        proposed = self._b
        if abs(slack) > self.deadband:
            proposed = self._b + self.gain * 8.0 * slack
        self._b = _clamp(min(proposed, self._kv_ceiling(obs)),
                         self.batch_min, self.batch_max)
        self.max_num_seqs = int(round(self._b))
        return ControlAction(max_num_seqs=self.max_num_seqs)


class TTFTSlackAdmit(Controller):
    """Admission driven by TTFT slack alone (the queue-collapse lever).

    ``SlackAdmit`` above closes on TPOT slack (decode-side). Axis-1/Axis-2's
    ``regime`` classification (specloop_rt/analysis.py) treats TTFT-p99 breach
    (queueing/admission) and TPOT-p99 breach (decode) as distinct failure
    modes, and the sweep data shows the admission cap moving the *SLO*
    numbers (queueing) far more than gamma ever moved the decode numbers. So
    this loop senses the queueing signal directly: shrink the cap when TTFT is
    already eating into its SLO budget (stop admitting into a queue that's
    collapsing), grow it back when there is slack. Deliberately has no KV term
    of its own -- ``KVAwareAdmit`` composes this with a KV ceiling so the two
    failure modes (queue collapse vs. KV/preemption thrashing) stay
    attributable to separate signals instead of one controller conflating them.

    Reacts to the queue *trend*, not just its level -- see
    ``_TrendAwareQueueTerm``: a queue that is large but draining loosens the
    cap sooner than one that is large and still growing, instead of both
    waiting for the same level threshold.
    """
    name = "ttft-slack-admit"

    def __init__(self, target_util: float = 0.7, gain: float = 0.5,
                 deadband: float = 0.05, period: int = 4,
                 batch_min: int = 1, batch_max: int = 256, init: int = 64,
                 trend_gain: float = 0.5, trend_ema_beta: float = 0.3):
        self.target_util = target_util
        self.gain = gain
        self.deadband = deadband
        self.period = max(1, period)
        self.batch_min, self.batch_max = batch_min, batch_max
        self._b = float(init)
        self.max_num_seqs = init
        self._queue_term = _TrendAwareQueueTerm(target_util, trend_gain, trend_ema_beta)

    def reset(self):
        pass

    def _proposed_cap(self, obs) -> float:
        slack = self._queue_term.slack(obs, obs.max_num_seqs_current)
        return self._b + self.gain * 8.0 * slack if abs(slack) > self.deadband else self._b

    def on_step(self, obs):
        if obs.step % self.period != 0:
            return ControlAction(max_num_seqs=self.max_num_seqs)
        self._b = _clamp(self._proposed_cap(obs), self.batch_min, self.batch_max)
        self.max_num_seqs = int(round(self._b))
        return ControlAction(max_num_seqs=self.max_num_seqs)


class KVAwareAdmit(Controller):
    """Admission-primary controller: TTFT-slack drives the cap, KV headroom
    bounds it from above.

    This is the primary actuator of the admit-primary/gamma-secondary design
    (see specloop_rt/README.md and the Axis-3 sweep this backs). Two terms,
    combined as ``min``, not a blend, because they answer different questions
    and a blend would let one mask the other exactly when isolating them
    matters most for the bracketing arms:

      * ``ttft_term`` -- same queueing-slack law as ``TTFTSlackAdmit``: shrink
        the cap when the queue is backing up, grow it back when there is
        slack. This is the lever Axis-1/Axis-2 showed has real leverage on
        SLO attainment and TTFT.
      * ``kv_ceiling`` -- a proportional cap on top of ``kv_used_frac``, not a
        single-threshold clamp like ``SlackAdmit``'s ``if kv_used_frac > 0.92``
        guard. Above ``kv_target``, the ceiling falls off linearly toward
        ``batch_min`` as kv_used_frac approaches 1.0, so admission tightens
        progressively as KV pressure builds instead of doing nothing until a
        single trip-wire, then slamming down. The goal is to prevent the
        preemption/recompute thrashing that (per the KV-accounting note in
        specloop/README.md's cost model) is what actually wrecks e2e-p95 and
        ITL -- no gamma policy touches this, because gamma is not what is
        being preempted, occupied KV blocks are.

    ``final = min(ttft_term, kv_ceiling)``: KV headroom is a hard safety bound
    that TTFT slack is never allowed to override, matching the stated design
    ("bound the cap from above by KV headroom ... regardless of TTFT slack").

    ``ttft_term`` is trend-aware, same as ``TTFTSlackAdmit`` -- see
    ``_TrendAwareQueueTerm``.
    """
    name = "kv-aware-admit"

    def __init__(self, target_util: float = 0.7, gain: float = 0.5,
                 deadband: float = 0.05, period: int = 4,
                 kv_target: float = 0.80, kv_min_frac: float = 0.30,
                 batch_min: int = 1, batch_max: int = 256, init: int = 64,
                 trend_gain: float = 0.5, trend_ema_beta: float = 0.3):
        self.target_util = target_util
        self.gain = gain
        self.deadband = deadband
        self.period = max(1, period)
        self.kv_target = kv_target
        self.kv_min_frac = kv_min_frac
        self.batch_min, self.batch_max = batch_min, batch_max
        self._b = float(init)
        self.max_num_seqs = init
        self._queue_term = _TrendAwareQueueTerm(target_util, trend_gain, trend_ema_beta)

    def reset(self):
        pass

    def _ttft_term(self, obs) -> float:
        slack = self._queue_term.slack(obs, obs.max_num_seqs_current)
        return self._b + self.gain * 8.0 * slack if abs(slack) > self.deadband else self._b

    def _kv_ceiling(self, obs) -> float:
        kv = obs.kv_used_frac
        if kv <= self.kv_target:
            return self.batch_max
        # linear falloff from batch_max at kv_target to kv_min_frac*batch_max
        # at kv_used_frac=1.0, so tightening is proportional to how far past
        # the target KV occupancy already is, not a single step function.
        over = (kv - self.kv_target) / max(1e-9, 1.0 - self.kv_target)
        floor = self.kv_min_frac * self.batch_max
        return self.batch_max - over * (self.batch_max - floor)

    def on_step(self, obs):
        if obs.step % self.period != 0:
            return ControlAction(max_num_seqs=self.max_num_seqs)
        ttft_term = self._ttft_term(obs)
        kv_ceiling = self._kv_ceiling(obs)
        self._b = _clamp(min(ttft_term, kv_ceiling), self.batch_min, self.batch_max)
        self.max_num_seqs = int(round(self._b))
        return ControlAction(max_num_seqs=self.max_num_seqs)


ADMIT_CONTROLLERS = {
    "static": StaticAdmit,
    "slack": SlackAdmit,
    "ttft-slack": TTFTSlackAdmit,
    "kv-aware": KVAwareAdmit,
    "ttft-predictive": TTFTPredictiveAdmit,
    "kv-predictive": KVPredictiveAdmit,
}


# ==========================================================================
# Composite: run a spec controller and an admit controller under a coordinator
# ==========================================================================


class Composite(Controller):
    """Combines an L_spec and an L_admit under a coordination policy.

    coordination:
      * "naive"      — both act on their own periods, no awareness
      * "timescale"  — admit forced onto a k-times-slower clock
      * "hysteresis" — a loop may not act within cooldown of the other's action
    """
    name = "composite"

    def __init__(self, spec: Controller, admit: Controller,
                 coordination: str = "naive", ratio: int = 10,
                 cooldown: int = 20, deadband: float = 0.1):
        self.spec = spec
        self.admit = admit
        self.coordination = coordination
        self.ratio = ratio
        self.cooldown = cooldown
        self.deadband = deadband
        self._last_spec = -10 ** 9
        self._last_admit = -10 ** 9

    def reset(self):
        self.spec.reset()
        self.admit.reset()

    def on_step(self, obs):
        obs.last_spec_action_step = self._last_spec
        obs.last_admit_action_step = self._last_admit
        g = obs.gamma_current
        mb = obs.max_num_seqs_current

        if self.coordination == "timescale":
            sa = self.spec.on_step(obs)
            g = sa.gamma if sa.gamma is not None else g
            self._last_spec = obs.step
            slow = getattr(self.spec, "period", 1) * self.ratio
            if obs.step % max(1, slow) == 0:
                aa = self.admit.on_step(obs)
                mb = aa.max_num_seqs if aa.max_num_seqs is not None else mb
                self._last_admit = obs.step
            return ControlAction(gamma=g, max_num_seqs=mb)

        if self.coordination == "hysteresis":
            if obs.step - self._last_admit > self.cooldown:
                sa = self.spec.on_step(obs)
                if sa.gamma is not None and abs(sa.gamma - g) / max(1, g) > self.deadband:
                    g = sa.gamma
                    self._last_spec = obs.step
            if obs.step - self._last_spec > self.cooldown:
                aa = self.admit.on_step(obs)
                if aa.max_num_seqs is not None and abs(aa.max_num_seqs - mb) / max(1, mb) > self.deadband:
                    mb = aa.max_num_seqs
                    self._last_admit = obs.step
            return ControlAction(gamma=g, max_num_seqs=mb)

        # naive
        sa = self.spec.on_step(obs)
        aa = self.admit.on_step(obs)
        if sa.gamma is not None:
            g = sa.gamma
            self._last_spec = obs.step
        if aa.max_num_seqs is not None:
            mb = aa.max_num_seqs
            self._last_admit = obs.step
        return ControlAction(gamma=g, max_num_seqs=mb,
                             per_request_gamma=sa.per_request_gamma)


def build_controller(spec_cfg: dict, tpot_slo: float) -> Controller:
    """Factory from a config dict (see configs/*.yaml)."""
    s = spec_cfg
    spec = SPEC_CONTROLLERS[s["spec"]](**s.get("spec_kw", {}))
    admit = ADMIT_CONTROLLERS[s["admit"]](**s.get("admit_kw", {}))
    return Composite(spec, admit, coordination=s.get("coordination", "naive"),
                     **s.get("coord_kw", {}))
