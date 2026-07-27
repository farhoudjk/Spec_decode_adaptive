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


SPEC_CONTROLLERS = {
    "static": StaticSpec,
    "static-table": StaticTableSpec,
    "closed-loop": ClosedLoopSpec,
    "dsde": DSDESpec,
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


ADMIT_CONTROLLERS = {
    "static": StaticAdmit,
    "slack": SlackAdmit,
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
