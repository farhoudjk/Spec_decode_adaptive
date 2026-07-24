"""Step-level LLM serving simulator with speculative decoding.

Why a simulator: RQ2 needs a factorial sweep over (gain x period-ratio x
volatility x controller), which is O(10^3-10^4) runs.  Real-hardware runs are
reserved for RQ3/RQ5 confirmation of the regions the sweep flags.  The
controller API here is deliberately identical to what a vLLM integration would
expose (``Observation`` in, actuation out), so the same controller objects can
drive either backend.

Cost model (roofline, per decode step):
    t_mem  = (W_target + KV_batch_bytes) / BW_eff
    t_comp = 2 * P_active * Q_tokens / FLOPS_eff
    t_step = max(t_mem, t_comp) + t_draft + overhead

Q_tokens = sum_i (gamma_i + 1) over the running batch, plus chunked-prefill
tokens.  This is what makes speculation *not* free at high batch width: raising
gamma raises Q_tokens, which pushes the step from the memory-bound regime into
the compute-bound regime.  That transition is the physical mechanism the
admission controller reacts to, and hence the mechanism of the loop coupling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .workload import HardwareConfig, ModelConfig, Request, WorkloadSpec


@dataclass
class Observation:
    """What controllers see. Identical shape in sim and in a vLLM integration."""
    t: float
    step: int
    running: List[Request]
    queue_len: int
    kv_used_frac: float
    batch_size: int
    # rolling signals
    accepted_ema: float          # mean accepted tokens per request per step
    alpha_ema: float             # mean per-token acceptance
    step_time_ema: float
    itl_p95: float
    tpot_slo_s: float
    ttft_slo_s: float
    gamma_current: int
    max_batch_current: int
    # last actuation by the *other* loop (used by coordination mechanisms)
    last_spec_action_step: int = -10 ** 9
    last_admit_action_step: int = -10 ** 9


@dataclass
class StepRecord:
    t: float
    step: int
    gamma: float
    batch_size: int
    max_batch: int
    queue_len: int
    kv_used_frac: float
    step_time: float
    accepted_mean: float
    alpha_mean: float
    tokens_out: int
    admitted: int
    preempted: int
    goodput_tok_s: float


@dataclass
class SimConfig:
    hardware: str = "H100-80G"
    model: str = "llama3.2-3b/1b"
    kv_capacity_frac: float = 0.85     # fraction of HBM left for KV after weights
    max_batch_hard: int = 256
    tpot_slo_s: float = 0.050
    ttft_slo_s: float = 2.0
    chunked_prefill_budget: int = 2048
    step_overhead_s: float = 0.0012
    ema_beta: float = 0.2
    max_steps: int = 200_000
    seed: int = 0
    # If a request is preempted this many times it is dropped (pathology guard)
    max_preemptions: int = 8
    # Hard cap on simulated horizon.  Under overload the backlog drain dominates
    # wall-clock, so measurement is bounded to a steady-state window instead.
    max_sim_s: Optional[float] = None


class Simulator:
    def __init__(self, cfg: SimConfig, hw: HardwareConfig, model: ModelConfig,
                 workload: WorkloadSpec, spec_ctrl, admit_ctrl, coordinator=None):
        self.cfg = cfg
        self.hw = hw
        self.model = model
        self.requests = workload.generate()
        self.spec_ctrl = spec_ctrl
        self.admit_ctrl = admit_ctrl
        self.coordinator = coordinator
        self.rng = np.random.default_rng(cfg.seed)

        weight_gb = (model.target_weight_bytes + model.draft_weight_bytes) / 1e9
        free_gb = max(1.0, hw.hbm_capacity_gb * hw.num_gpus - weight_gb)
        self.kv_capacity_bytes = free_gb * 1e9 * cfg.kv_capacity_frac
        self.kv_bytes_per_token = model.kv_bytes_per_token + model.draft_kv_bytes_per_token

        self.records: List[StepRecord] = []
        self.finished: List[Request] = []
        self.dropped: int = 0

    # ------------------------------------------------------------------
    # cost model
    # ------------------------------------------------------------------
    def _kv_bytes(self, running: List[Request], gammas: Dict[int, int]) -> float:
        """Resident KV + slots *reserved* for speculative tokens before accept."""
        total = 0.0
        for r in running:
            total += (r.seq_len + gammas.get(r.rid, 0)) * self.kv_bytes_per_token
        return total

    def _kv_of(self, seq_sum: int, gamma_sum: int) -> float:
        """O(1) form of _kv_bytes for the inner admission/preemption loops."""
        return (seq_sum + gamma_sum) * self.kv_bytes_per_token

    def _step_time(self, running: List[Request], gammas: Dict[int, int],
                   prefill_tokens: int) -> float:
        m = self.model
        kv_bytes = self._kv_bytes(running, gammas)
        t_mem = (m.target_weight_bytes + kv_bytes) / self.hw.eff_bw_bytes_s

        q_tokens = sum(gammas.get(r.rid, 0) + 1 for r in running) + prefill_tokens
        flops = 2.0 * m.active_params_b * 1e9 * q_tokens
        if m.is_moe and q_tokens > 0:
            # routing imbalance: the slowest expert paces the step
            flops *= (1.0 + m.moe_imbalance)
        t_comp = flops / self.hw.eff_flops_s

        t_verify = max(t_mem, t_comp)

        # draft phase
        t_draft = 0.0
        gmax = max(gammas.values()) if gammas else 0
        if gmax > 0:
            if m.draft_mode == "ngram":
                t_draft = 2e-5 * gmax                       # table lookup
            elif m.draft_mode == "self_spec":
                d_mem = 0.05 * m.target_weight_bytes / self.hw.eff_bw_bytes_s
                t_draft = d_mem + 0.02 * t_verify           # one fused cheap pass
            else:
                d_kv = sum((r.seq_len) * m.draft_kv_bytes_per_token for r in running)
                d_mem = (m.draft_weight_bytes + d_kv) / self.hw.eff_bw_bytes_s
                d_flops = 2.0 * m.draft_active_params_b * 1e9 * len(running)
                d_comp = d_flops / self.hw.eff_flops_s
                t_draft = gmax * (max(d_mem, d_comp) + 0.15 * self.cfg.step_overhead_s)

        return t_verify + t_draft + self.cfg.step_overhead_s

    # ------------------------------------------------------------------
    def _sample_accepted(self, r: Request, gamma: int) -> int:
        """Leading-run acceptance: P(n=j)=a^j(1-a) for j<gamma, P(n=gamma)=a^gamma."""
        if gamma <= 0:
            return 0
        a = r.current_alpha()
        u = self.rng.random(gamma)
        n = int(np.argmax(u > a)) if np.any(u > a) else gamma
        return n

    # ------------------------------------------------------------------
    def run(self) -> "SimResult":
        cfg = self.cfg
        pending = list(self.requests)
        pi = 0
        queue: List[Request] = []
        running: List[Request] = []
        t = 0.0
        step = 0
        b = cfg.ema_beta
        accepted_ema = 1.0
        alpha_ema = 0.6
        step_time_ema = 0.01
        recent_itl: List[float] = []
        tokens_emitted = 0
        last_spec_step = -10 ** 9
        last_admit_step = -10 ** 9

        gamma_global = self.spec_ctrl.initial_gamma()
        max_batch = self.admit_ctrl.initial_max_batch()

        horizon = self.requests[-1].arrival_s if self.requests else 0.0

        while step < cfg.max_steps:
            # ---- arrivals -------------------------------------------------
            while pi < len(pending) and pending[pi].arrival_s <= t:
                queue.append(pending[pi])
                pi += 1
            if not queue and not running and pi >= len(pending):
                break
            if not queue and not running:
                t = pending[pi].arrival_s
                continue

            itl_p95 = float(np.percentile(recent_itl[-2000:], 95)) if recent_itl else 0.0
            obs = Observation(
                t=t, step=step, running=running, queue_len=len(queue),
                kv_used_frac=self._kv_bytes(running, {}) / self.kv_capacity_bytes,
                batch_size=len(running), accepted_ema=accepted_ema,
                alpha_ema=alpha_ema, step_time_ema=step_time_ema, itl_p95=itl_p95,
                tpot_slo_s=cfg.tpot_slo_s, ttft_slo_s=cfg.ttft_slo_s,
                gamma_current=gamma_global, max_batch_current=max_batch,
                last_spec_action_step=last_spec_step,
                last_admit_action_step=last_admit_step,
            )

            # ---- control --------------------------------------------------
            if self.coordinator is not None:
                gamma_global, max_batch, acted = self.coordinator.update(
                    obs, self.spec_ctrl, self.admit_ctrl)
                if acted.get("spec"):
                    last_spec_step = step
                if acted.get("admit"):
                    last_admit_step = step
            else:
                if self.spec_ctrl.should_act(step):
                    gamma_global = self.spec_ctrl.update(obs)
                    last_spec_step = step
                if self.admit_ctrl.should_act(step):
                    max_batch = self.admit_ctrl.update(obs)
                    last_admit_step = step
            max_batch = int(np.clip(max_batch, 1, cfg.max_batch_hard))

            # ---- admission ------------------------------------------------
            admitted = 0
            seq_sum = sum(r.seq_len for r in running)
            while queue and len(running) < max_batch:
                cand = queue[0]
                if self._kv_of(seq_sum + cand.seq_len, 0) > self.kv_capacity_bytes:
                    break
                queue.pop(0)
                cand.admitted_s = t
                running.append(cand)
                seq_sum += cand.seq_len
                admitted += 1

            # ---- per-request gamma (ragged Q if the controller supports it)
            gammas: Dict[int, int] = {}
            for r in running:
                if not r.prefill_done:
                    gammas[r.rid] = 0
                else:
                    gammas[r.rid] = self.spec_ctrl.gamma_for(r, gamma_global, obs)

            # ---- KV admission-control on speculative reservations ---------
            # If speculation would overflow KV, shrink gammas before preempting.
            gamma_sum = sum(gammas.values())
            while self._kv_of(seq_sum, gamma_sum) > self.kv_capacity_bytes:
                g_max = max(gammas.values()) if gammas else 0
                if g_max <= 0:
                    break
                for rid, g in gammas.items():
                    if g == g_max:
                        gammas[rid] = g - 1
                        gamma_sum -= 1

            # ---- preemption (KV pressure) ---------------------------------
            preempted = 0
            while running and self._kv_of(seq_sum, gamma_sum) > self.kv_capacity_bytes:
                victim = max(running, key=lambda r: r.seq_len)
                running.remove(victim)
                seq_sum -= victim.seq_len
                gamma_sum -= gammas.pop(victim.rid, 0)
                victim.preempted_count += 1
                victim.generated = 0
                victim.prefill_done = False
                victim.admitted_s = None
                preempted += 1
                if victim.preempted_count > cfg.max_preemptions:
                    self.dropped += 1
                else:
                    queue.insert(0, victim)

            # ---- chunked prefill ------------------------------------------
            prefill_tokens = 0
            budget = cfg.chunked_prefill_budget
            for r in running:
                if r.prefill_done or budget <= 0:
                    continue
                take = min(budget, r.prompt_len)
                prefill_tokens += take
                budget -= take
                r.prefill_done = True   # single-shot prefill approximation

            # ---- execute step ---------------------------------------------
            dt = self._step_time(running, gammas, prefill_tokens)
            step_tokens = 0
            acc_list: List[int] = []
            alpha_list: List[float] = []
            done: List[Request] = []
            for r in running:
                g = gammas.get(r.rid, 0)
                if not r.prefill_done:
                    continue
                n_acc = self._sample_accepted(r, g)
                n_tok = n_acc + 1                      # + corrected/bonus token
                n_tok = min(n_tok, r.output_len - r.generated)
                if n_tok <= 0:
                    n_tok = 1
                alpha_list.append(r.current_alpha())
                acc_list.append(n_acc)
                r.gamma_seen.append(g)
                r.accepted_seen.append(n_acc)
                r.generated += n_tok
                step_tokens += n_tok
                if r.first_token_s is None:
                    r.first_token_s = t + dt
                per_tok = dt / max(1, n_tok)
                r.itls.extend([per_tok] * n_tok)
                recent_itl.extend([per_tok] * n_tok)
                if r.generated >= r.output_len:
                    r.finished_s = t + dt
                    done.append(r)

            for r in done:
                running.remove(r)
                self.finished.append(r)

            tokens_emitted += step_tokens
            acc_mean = float(np.mean(acc_list)) if acc_list else 0.0
            alpha_mean = float(np.mean(alpha_list)) if alpha_list else alpha_ema
            accepted_ema = (1 - b) * accepted_ema + b * acc_mean
            alpha_ema = (1 - b) * alpha_ema + b * alpha_mean
            step_time_ema = (1 - b) * step_time_ema + b * dt

            self.records.append(StepRecord(
                t=t, step=step, gamma=float(np.mean(list(gammas.values())) if gammas else 0),
                batch_size=len(running) + len(done), max_batch=max_batch,
                queue_len=len(queue), kv_used_frac=self._kv_bytes(running, {}) / self.kv_capacity_bytes,
                step_time=dt, accepted_mean=acc_mean, alpha_mean=alpha_mean,
                tokens_out=step_tokens, admitted=admitted, preempted=preempted,
                goodput_tok_s=step_tokens / dt if dt > 0 else 0.0,
            ))

            t += dt
            step += 1
            if pi >= len(pending) and not queue and not running:
                break
            if cfg.max_sim_s is not None and t > cfg.max_sim_s:
                break
            if t > horizon * 6 + 600:      # runaway guard
                break

        return SimResult(cfg=self.cfg, model=self.model, hardware=self.hw,
                         records=self.records, finished=self.finished,
                         dropped=self.dropped, wall_s=t, total_steps=step,
                         n_submitted=len(self.requests))


@dataclass
class SimResult:
    cfg: SimConfig
    model: ModelConfig
    hardware: HardwareConfig
    records: List[StepRecord]
    finished: List[Request]
    dropped: int
    wall_s: float
    total_steps: int
    n_submitted: int

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame([r.__dict__ for r in self.records])
