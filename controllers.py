"""Controllers and coordination mechanisms.

Two independent loops:
  L_spec  : senses acceptance statistics, actuates gamma (speculation length)
  L_admit : senses latency/queue vs SLO, actuates the admitted batch cap

Every controller exposes the same four control-theoretic knobs used by RQ2:
  period (control interval in steps), gain (reaction aggressiveness),
  a sensing signal, and an actuation variable.

The SOTA L_spec variants below are re-implementations of the *published control
laws*, not of the full systems.  Anything reported from them must be described
as a faithful reimplementation of the controller, not a reproduction of the
original paper's end-to-end system.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# ==========================================================================
# Base
# ==========================================================================


class SpecController:
    name = "base"

    def __init__(self, period: int = 1, gain: float = 1.0, gamma_min: int = 0,
                 gamma_max: int = 10, gamma_init: int = 4):
        self.period = max(1, int(period))
        self.gain = float(gain)
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.gamma = gamma_init
        # continuous internal state; only the actuation is quantized.  Rounding
        # the accumulator each step silently freezes the loop at small gains.
        self._gamma_f = float(gamma_init)
        self.adaptive = True

    def initial_gamma(self) -> int:
        return self.gamma

    def should_act(self, step: int) -> bool:
        return self.adaptive and (step % self.period == 0)

    def update(self, obs) -> int:
        raise NotImplementedError

    def gamma_for(self, req, gamma_global: int, obs) -> int:
        """Per-request gamma. Uniform unless the controller is ragged (DSDE)."""
        return int(gamma_global)


class AdmitController:
    name = "base"

    def __init__(self, period: int = 1, gain: float = 1.0, max_batch_init: int = 64,
                 batch_min: int = 1, batch_max: int = 256):
        self.period = max(1, int(period))
        self.gain = float(gain)
        self.max_batch = max_batch_init
        self._batch_f = float(max_batch_init)
        self.batch_min = batch_min
        self.batch_max = batch_max
        self.adaptive = True

    def initial_max_batch(self) -> int:
        return self.max_batch

    def should_act(self, step: int) -> bool:
        return self.adaptive and (step % self.period == 0)

    def update(self, obs) -> int:
        raise NotImplementedError


# ==========================================================================
# L_spec variants
# ==========================================================================


class StaticSpec(SpecController):
    """No speculation control. gamma fixed (gamma=0 => speculation disabled)."""
    name = "static-spec"

    def __init__(self, gamma: int = 4, **kw):
        kw.pop("gamma_init", None)
        super().__init__(period=10 ** 9, gamma_init=gamma, **kw)
        self.adaptive = False

    def update(self, obs) -> int:
        return self.gamma


class EMASpec(SpecController):
    """SmartSpec/DSDE-style: raise gamma while marginal acceptance stays high.

    Control law: track EMA of accepted length; the marginal benefit of the next
    speculative token is approximately alpha^gamma.  Step gamma up when that
    exceeds a threshold, down otherwise.  ``gain`` scales the step size.
    """
    name = "ema-spec"

    def __init__(self, threshold: float = 0.45, deadband: float = 0.5, **kw):
        super().__init__(**kw)
        self.threshold = threshold
        self.deadband = deadband

    def update(self, obs) -> int:
        # setpoint: largest gamma whose marginal acceptance a^gamma still clears
        # the threshold.  Proportional move toward it, with a deadband so the
        # loop SETTLES in isolation rather than chattering (a bang-bang law
        # would self-oscillate and confound the interference attribution).
        a = max(1e-3, min(0.999, obs.alpha_ema))
        target = np.log(self.threshold) / np.log(a)
        err = target - self.gamma
        if abs(err) <= self.deadband:
            return self.gamma
        delta = np.clip(self.gain * err, -self.gain * 2.0, self.gain * 2.0)
        self._gamma_f = float(np.clip(self._gamma_f + delta, self.gamma_min, self.gamma_max))
        self.gamma = int(round(self._gamma_f))
        return self.gamma


class BanditSpec(SpecController):
    """AdaSpec-style: UCB over discrete gamma arms, reward = observed goodput."""
    name = "bandit-spec"

    def __init__(self, c: float = 0.6, **kw):
        super().__init__(**kw)
        self.arms = list(range(self.gamma_min, self.gamma_max + 1))
        self.n = np.zeros(len(self.arms))
        self.q = np.zeros(len(self.arms))
        self.c = c
        self.t = 0
        self._last_arm = self.arms.index(int(np.clip(self.gamma, self.gamma_min, self.gamma_max)))

    def update(self, obs) -> int:
        # reward: tokens per second achieved since last action
        reward = (obs.accepted_ema + 1.0) / max(obs.step_time_ema, 1e-6)
        i = self._last_arm
        self.n[i] += 1
        self.q[i] += (reward - self.q[i]) / self.n[i]
        self.t += 1
        with np.errstate(divide="ignore", invalid="ignore"):
            bonus = self.c * self.gain * np.sqrt(np.log(max(self.t, 2)) / np.maximum(self.n, 1e-9))
        bonus[self.n == 0] = 1e9
        scale = max(np.max(np.abs(self.q)), 1e-9)
        j = int(np.argmax(self.q / scale + bonus))
        self._last_arm = j
        self.gamma = self.arms[j]
        return self.gamma


class EntropySpec(SpecController):
    """HeteroSpec-style: bin a predictability proxy, map bin -> gamma.

    The published metric is top-K entropy of the draft distribution; here the
    simulator exposes (1 - alpha) as the equivalent unpredictability proxy.
    """
    name = "entropy-spec"

    def __init__(self, bins=(0.15, 0.30, 0.45), **kw):
        super().__init__(**kw)
        self.bins = bins

    def update(self, obs) -> int:
        u = 1.0 - obs.alpha_ema
        b = int(np.digitize(u, self.bins))
        table = [self.gamma_max, int(0.7 * self.gamma_max), int(0.4 * self.gamma_max), self.gamma_min + 1]
        target = table[min(b, len(table) - 1)]
        self._gamma_f = float(np.clip(self._gamma_f + self.gain * np.sign(target - self._gamma_f),
                                      self.gamma_min, self.gamma_max))
        self.gamma = int(round(self._gamma_f))
        return self.gamma


class DSDESpec(SpecController):
    """DSDE-style per-sequence (ragged) speculation length.

    Global gamma is the batch mean; each request gets its own gamma from its own
    acceptance EMA.  This is the variant most likely to interact with admission
    control, because the *composition* of the batch now directly sets the work
    per step.
    """
    name = "dsde-spec"

    def __init__(self, **kw):
        super().__init__(**kw)

    def update(self, obs) -> int:
        gs = [r.gamma for r in obs.running if r.gamma > 0]
        self.gamma = int(np.clip(round(np.mean(gs)) if gs else self.gamma,
                                 self.gamma_min, self.gamma_max))
        return self.gamma

    def gamma_for(self, req, gamma_global: int, obs) -> int:
        if req.accepted_seen:
            last_g = req.gamma_seen[-1] if req.gamma_seen else max(1, gamma_global)
            hit = req.accepted_seen[-1] / max(1, last_g)
            req.alpha_ema = 0.7 * req.alpha_ema + 0.3 * hit if req.alpha_ema > 0 else hit
        a = max(1e-3, min(0.999, req.alpha_ema if req.alpha_ema > 0 else obs.alpha_ema))
        # setpoint: extend while marginal acceptance a^g still exceeds 0.5.
        # gain smooths the approach; it must NOT scale the setpoint itself
        # (multiplying then truncating drives gamma to 0 for gain < 1).
        target = np.log(0.5) / np.log(a) if a < 0.999 else float(self.gamma_max)
        prev = float(req.gamma) if req.gamma > 0 else target
        smoothed = prev + self.gain * (target - prev)
        g = int(np.clip(round(smoothed), max(1, self.gamma_min), self.gamma_max))
        req.gamma = g
        return g


SPEC_CONTROLLERS = {
    "static": StaticSpec,
    "ema": EMASpec,
    "bandit": BanditSpec,
    "entropy": EntropySpec,
    "dsde": DSDESpec,
}


# ==========================================================================
# L_admit variants
# ==========================================================================


class StaticAdmit(AdmitController):
    """Fixed batch cap: the non-adaptive admission baseline."""
    name = "static-admit"

    def __init__(self, max_batch: int = 64, **kw):
        kw.pop("max_batch_init", None)
        super().__init__(period=10 ** 9, max_batch_init=max_batch, **kw)
        self.adaptive = False

    def update(self, obs) -> int:
        return self.max_batch


class SlackAdmit(AdmitController):
    """Slack-guided admission (Kairos/Chiron-style).

    Sense: measured per-token latency vs the TPOT SLO.  Actuate: batch cap.
    Note the controller has *no model* of gamma - it only sees the step time
    that gamma produces.  That blindness is the point of the study.
    """
    name = "slack-admit"

    def __init__(self, target_util: float = 0.85, deadband: float = 0.05, **kw):
        super().__init__(**kw)
        self.target_util = target_util
        self.deadband = deadband

    def update(self, obs) -> int:
        tpot_est = obs.step_time_ema / max(1e-6, obs.accepted_ema + 1.0)
        slack = (obs.tpot_slo_s * self.target_util - tpot_est) / max(obs.tpot_slo_s, 1e-9)
        if abs(slack) <= self.deadband:
            return self.max_batch
        delta = self.gain * 8.0 * slack
        if obs.kv_used_frac > 0.92:
            delta = min(delta, -self.gain * 4.0)
        self._batch_f = float(np.clip(self._batch_f + delta, self.batch_min, self.batch_max))
        self.max_batch = int(round(self._batch_f))
        return self.max_batch


class QueueAdmit(AdmitController):
    """Queue-length + SLO admission: reacts to backlog as well as latency."""
    name = "queue-admit"

    def __init__(self, **kw):
        super().__init__(**kw)

    def update(self, obs) -> int:
        tpot_est = obs.step_time_ema / max(1e-6, obs.accepted_ema + 1.0)
        over = tpot_est > obs.tpot_slo_s
        if over:
            delta = -self.gain * 6.0
        else:
            delta = self.gain * min(6.0, 0.5 * obs.queue_len)
        if obs.kv_used_frac > 0.92:
            delta = min(delta, -self.gain * 4.0)
        self._batch_f = float(np.clip(self._batch_f + delta, self.batch_min, self.batch_max))
        self.max_batch = int(round(self._batch_f))
        return self.max_batch


ADMIT_CONTROLLERS = {
    "static": StaticAdmit,
    "slack": SlackAdmit,
    "queue": QueueAdmit,
}


# ==========================================================================
# Coordination mechanisms (RQ4)
# ==========================================================================


class Coordinator:
    name = "base"

    def update(self, obs, spec_ctrl, admit_ctrl):
        raise NotImplementedError


class NaiveComposition(Coordinator):
    """Both loops act on their own periods with no awareness of each other."""
    name = "naive"

    def update(self, obs, spec_ctrl, admit_ctrl):
        acted = {"spec": False, "admit": False}
        g, mb = obs.gamma_current, obs.max_batch_current
        if spec_ctrl.should_act(obs.step):
            g = spec_ctrl.update(obs)
            acted["spec"] = True
        if admit_ctrl.should_act(obs.step):
            mb = admit_ctrl.update(obs)
            acted["admit"] = True
        return g, mb, acted


class TimescaleSeparation(Coordinator):
    """Force a k-fold timescale gap: fast inner L_spec, slow supervisory L_admit.

    Classical singular-perturbation fix. L_admit additionally sees a low-pass
    filtered latency signal so it cannot chase L_spec's transients.
    """
    name = "timescale"

    def __init__(self, ratio: int = 10, filter_beta: float = 0.05):
        self.ratio = ratio
        self.filter_beta = filter_beta
        self._filt = None

    def update(self, obs, spec_ctrl, admit_ctrl):
        acted = {"spec": False, "admit": False}
        g, mb = obs.gamma_current, obs.max_batch_current
        if obs.step % max(1, spec_ctrl.period) == 0:
            g = spec_ctrl.update(obs)
            acted["spec"] = True
        slow_period = max(1, spec_ctrl.period * self.ratio)
        if obs.step % slow_period == 0:
            if self._filt is None:
                self._filt = obs.step_time_ema
            self._filt = (1 - self.filter_beta) * self._filt + self.filter_beta * obs.step_time_ema
            shadow = _clone_obs(obs, step_time_ema=self._filt)
            mb = admit_ctrl.update(shadow)
            acted["admit"] = True
        return g, mb, acted


class HysteresisCoordination(Coordinator):
    """Deadband + cooldown: a loop may not react inside the settling window of
    the other loop's most recent actuation."""
    name = "hysteresis"

    def __init__(self, cooldown_steps: int = 20, deadband: float = 0.10):
        self.cooldown = cooldown_steps
        self.deadband = deadband

    def update(self, obs, spec_ctrl, admit_ctrl):
        acted = {"spec": False, "admit": False}
        g, mb = obs.gamma_current, obs.max_batch_current
        since_admit = obs.step - obs.last_admit_action_step
        since_spec = obs.step - obs.last_spec_action_step

        if spec_ctrl.should_act(obs.step) and since_admit > self.cooldown:
            g_new = spec_ctrl.update(obs)
            if abs(g_new - g) / max(1, g) > self.deadband:
                g = g_new
                acted["spec"] = True
            else:
                spec_ctrl.gamma = g

        if admit_ctrl.should_act(obs.step) and since_spec > self.cooldown:
            mb_new = admit_ctrl.update(obs)
            if abs(mb_new - mb) / max(1, mb) > self.deadband:
                mb = mb_new
                acted["admit"] = True
            else:
                admit_ctrl.max_batch = mb
        return g, mb, acted


class MIMOCoordinator(Coordinator):
    """Single joint controller over (gamma, batch_cap).

    Instead of two SISO loops, do a one-step lookahead over the cross product
    using an analytic step-time/goodput model, subject to the TPOT SLO.  This is
    the "collapse the loops" arm of RQ4.
    """
    name = "mimo"

    def __init__(self, period: int = 4, gamma_max: int = 10, batch_max: int = 256,
                 batch_step: int = 8):
        self.period = period
        self.gamma_max = gamma_max
        self.batch_max = batch_max
        self.batch_step = batch_step

    def update(self, obs, spec_ctrl, admit_ctrl):
        acted = {"spec": False, "admit": False}
        g, mb = obs.gamma_current, obs.max_batch_current
        if obs.step % self.period != 0:
            return g, mb, acted

        a = float(np.clip(obs.alpha_ema, 1e-3, 0.999))
        bsz = max(1, obs.batch_size)
        t_unit = obs.step_time_ema / max(1.0, bsz * (obs.gamma_current + 1))

        best = (-1.0, g, mb)
        for gg in range(0, self.gamma_max + 1):
            exp_tok = (1 - a ** (gg + 1)) / (1 - a)      # Leviathan expected accepted+1
            for bb in range(self.batch_step, self.batch_max + 1, self.batch_step):
                eff_b = min(bb, bsz + obs.queue_len)
                if eff_b <= 0:
                    continue
                t_step = t_unit * eff_b * (gg + 1) + 1e-4
                tpot = t_step / exp_tok
                if tpot > obs.tpot_slo_s:
                    continue
                goodput = eff_b * exp_tok / t_step
                if goodput > best[0]:
                    best = (goodput, gg, bb)
        if best[0] > 0:
            _, g, mb = best
            spec_ctrl.gamma = g
            admit_ctrl.max_batch = mb
            acted = {"spec": True, "admit": True}
        return g, mb, acted


COORDINATORS = {
    "naive": NaiveComposition,
    "timescale": TimescaleSeparation,
    "hysteresis": HysteresisCoordination,
    "mimo": MIMOCoordinator,
}


def _clone_obs(obs, **overrides):
    from copy import copy
    o = copy(obs)
    for k, v in overrides.items():
        setattr(o, k, v)
    return o
