"""Model configs, hardware configs, and workload generation.

The acceptance model is the key coupling point of the whole study: a request's
acceptance rate alpha depends on (a) its workload type, (b) its generation
phase, and (c) the draft strategy.  Because the admission controller changes
*which* requests are resident in the batch, it changes the batch-average alpha,
which is exactly what the speculation controller senses.  That is the loop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------
# Hardware
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HardwareConfig:
    name: str
    hbm_bandwidth_gbs: float      # GB/s
    peak_flops_tflops: float      # dense TFLOP/s at serving dtype
    hbm_capacity_gb: float
    num_gpus: int = 1
    mfu: float = 0.45             # achieved fraction of peak on real kernels
    membw_eff: float = 0.80       # achieved fraction of peak bandwidth

    @property
    def eff_bw_bytes_s(self) -> float:
        return self.hbm_bandwidth_gbs * 1e9 * self.membw_eff * self.num_gpus

    @property
    def eff_flops_s(self) -> float:
        return self.peak_flops_tflops * 1e12 * self.mfu * self.num_gpus


HARDWARE = {
    "A100-80G": HardwareConfig("A100-80G", 2039, 312, 80),
    "H100-80G": HardwareConfig("H100-80G", 3350, 989, 80),
    "H100-80Gx4": HardwareConfig("H100-80Gx4", 3350, 989, 80, num_gpus=4),
}


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """Target/draft pair.

    ``active_params_b`` is what drives compute cost (differs from total for MoE).
    ``draft_mode`` selects the cost structure of the draft phase:
      - "separate": an independent small model, gamma sequential forward passes
      - "self_spec": EAGLE/Medusa-style heads on the target, ~one cheap pass
      - "ngram":     table lookup, ~zero GPU cost
    """
    name: str
    layers: int
    kv_heads: int
    head_dim: int
    total_params_b: float
    active_params_b: float
    dtype_bytes: int = 2
    kv_dtype_bytes: int = 2
    # draft side
    draft_mode: str = "separate"
    draft_active_params_b: float = 1.0
    draft_total_params_b: float = 1.0
    draft_layers: int = 16
    draft_kv_heads: int = 8
    draft_head_dim: int = 128
    is_moe: bool = False
    # MoE routing imbalance inflates the effective compute of a wide batch
    moe_imbalance: float = 0.0

    @property
    def kv_bytes_per_token(self) -> float:
        return 2 * self.layers * self.kv_heads * self.head_dim * self.kv_dtype_bytes

    @property
    def draft_kv_bytes_per_token(self) -> float:
        if self.draft_mode == "ngram":
            return 0.0
        if self.draft_mode == "self_spec":
            # EAGLE-style heads reuse the target features; small extra state only
            return 0.10 * self.kv_bytes_per_token
        return 2 * self.draft_layers * self.draft_kv_heads * self.draft_head_dim * self.kv_dtype_bytes

    @property
    def target_weight_bytes(self) -> float:
        return self.total_params_b * 1e9 * self.dtype_bytes

    @property
    def draft_weight_bytes(self) -> float:
        if self.draft_mode == "ngram":
            return 0.0
        if self.draft_mode == "self_spec":
            return 0.02 * self.target_weight_bytes
        return self.draft_total_params_b * 1e9 * self.dtype_bytes


MODELS: Dict[str, ModelConfig] = {
    # --- sweep-scale dense (cheap, used for RQ1/RQ2 large factorials) --------
    "llama3.2-3b/1b": ModelConfig(
        name="llama3.2-3b/1b", layers=28, kv_heads=8, head_dim=128,
        total_params_b=3.2, active_params_b=3.2,
        draft_mode="separate", draft_active_params_b=1.2, draft_total_params_b=1.2,
        draft_layers=16, draft_kv_heads=8, draft_head_dim=64,
    ),
    # --- production-scale dense (main result) -------------------------------
    "llama3.1-70b/8b": ModelConfig(
        name="llama3.1-70b/8b", layers=80, kv_heads=8, head_dim=128,
        total_params_b=70.0, active_params_b=70.0,
        draft_mode="separate", draft_active_params_b=8.0, draft_total_params_b=8.0,
        draft_layers=32, draft_kv_heads=8, draft_head_dim=128,
    ),
    # --- different GQA ratio / KV footprint ---------------------------------
    "qwen2.5-32b/1.5b": ModelConfig(
        name="qwen2.5-32b/1.5b", layers=64, kv_heads=8, head_dim=128,
        total_params_b=32.5, active_params_b=32.5,
        draft_mode="separate", draft_active_params_b=1.5, draft_total_params_b=1.5,
        draft_layers=28, draft_kv_heads=2, draft_head_dim=128,
    ),
    # --- MoE: active << total, routing imbalance ----------------------------
    "moe-8x7b/1b": ModelConfig(
        name="moe-8x7b/1b", layers=32, kv_heads=8, head_dim=128,
        total_params_b=46.7, active_params_b=12.9,
        draft_mode="separate", draft_active_params_b=1.1, draft_total_params_b=1.1,
        draft_layers=24, draft_kv_heads=8, draft_head_dim=128,
        is_moe=True, moe_imbalance=0.25,
    ),
    # --- self-speculative: no separate draft model --------------------------
    "llama3.1-8b/eagle": ModelConfig(
        name="llama3.1-8b/eagle", layers=32, kv_heads=8, head_dim=128,
        total_params_b=8.0, active_params_b=8.0,
        draft_mode="self_spec",
    ),
    # --- n-gram / prompt-lookup drafting ------------------------------------
    "llama3.1-8b/ngram": ModelConfig(
        name="llama3.1-8b/ngram", layers=32, kv_heads=8, head_dim=128,
        total_params_b=8.0, active_params_b=8.0,
        draft_mode="ngram",
    ),
}


# --------------------------------------------------------------------------
# Requests / workload
# --------------------------------------------------------------------------


@dataclass
class RequestType:
    """A class of traffic with its own acceptance behaviour.

    ``alpha_mu`` is the mean per-token acceptance probability under a *neural*
    draft model.  ``alpha_ngram_mu`` is the (usually very different) acceptance
    under prompt-lookup drafting: high for repetitive/RAG traffic, poor for
    open-ended chat.  ``alpha_decay`` models acceptance drifting over the course
    of generation, which is what makes a request's contribution to the batch
    signal non-stationary.
    """
    name: str
    prompt_mu: float
    prompt_sigma: float
    output_mu: float
    output_sigma: float
    alpha_mu: float
    alpha_sigma: float
    alpha_ngram_mu: float
    alpha_decay: float = 0.0
    weight: float = 1.0


REQUEST_TYPES: Dict[str, RequestType] = {
    # long prompt, short output, output echoes prompt -> n-gram loves this
    "rag": RequestType("rag", prompt_mu=7.6, prompt_sigma=0.5, output_mu=4.6,
                       output_sigma=0.6, alpha_mu=0.82, alpha_sigma=0.06,
                       alpha_ngram_mu=0.74, alpha_decay=0.02),
    # code completion: repetitive, structured
    "code": RequestType("code", prompt_mu=6.9, prompt_sigma=0.7, output_mu=5.3,
                        output_sigma=0.7, alpha_mu=0.78, alpha_sigma=0.07,
                        alpha_ngram_mu=0.62, alpha_decay=0.03),
    # open-ended chat: low n-gram acceptance, moderate neural acceptance
    "chat": RequestType("chat", prompt_mu=5.2, prompt_sigma=0.9, output_mu=6.0,
                        output_sigma=0.8, alpha_mu=0.63, alpha_sigma=0.09,
                        alpha_ngram_mu=0.21, alpha_decay=0.06),
    # long chain-of-thought reasoning: long output, acceptance decays
    "reason": RequestType("reason", prompt_mu=5.8, prompt_sigma=0.8, output_mu=7.0,
                          output_sigma=0.7, alpha_mu=0.68, alpha_sigma=0.08,
                          alpha_ngram_mu=0.28, alpha_decay=0.10),
}


@dataclass
class Request:
    rid: int
    arrival_s: float
    prompt_len: int
    output_len: int
    rtype: str
    alpha0: float
    alpha_decay: float
    # runtime state
    admitted_s: Optional[float] = None
    first_token_s: Optional[float] = None
    finished_s: Optional[float] = None
    generated: int = 0
    prefill_done: bool = False
    preempted_count: int = 0
    itls: List[float] = field(default_factory=list)
    gamma_seen: List[int] = field(default_factory=list)
    accepted_seen: List[int] = field(default_factory=list)
    # per-request controller state (DSDE-style ragged gamma)
    alpha_ema: float = 0.0
    gamma: int = 0

    @property
    def seq_len(self) -> int:
        return self.prompt_len + self.generated

    def current_alpha(self) -> float:
        """Acceptance decays as generation proceeds (phase drift)."""
        frac = self.generated / max(1, self.output_len)
        a = self.alpha0 * math.exp(-self.alpha_decay * frac * 3.0)
        return float(np.clip(a, 0.02, 0.98))


@dataclass
class WorkloadSpec:
    """Describes a synthetic arrival trace.

    ``phases`` is a list of (duration_s, mix_dict, rate_rps).  Switching mix
    mid-run is how RQ2's "workload volatility" axis is realised, and how the
    RQ1 step perturbation is injected.
    """
    name: str
    phases: List[tuple]
    seed: int = 0
    draft_kind: str = "neural"      # "neural" | "ngram"

    def generate(self) -> List[Request]:
        rng = np.random.default_rng(self.seed)
        reqs: List[Request] = []
        t = 0.0
        rid = 0
        for dur, mix, rate in self.phases:
            names = list(mix.keys())
            probs = np.array([mix[n] for n in names], dtype=float)
            probs = probs / probs.sum()
            t_end = t + dur
            while True:
                t = t + rng.exponential(1.0 / max(rate, 1e-9))
                if t >= t_end:
                    t = t_end
                    break
                rt = REQUEST_TYPES[names[rng.choice(len(names), p=probs)]]
                plen = int(np.clip(rng.lognormal(rt.prompt_mu, rt.prompt_sigma), 16, 32768))
                olen = int(np.clip(rng.lognormal(rt.output_mu, rt.output_sigma), 8, 4096))
                base = rt.alpha_ngram_mu if self.draft_kind == "ngram" else rt.alpha_mu
                a0 = float(np.clip(rng.normal(base, rt.alpha_sigma), 0.03, 0.97))
                reqs.append(Request(rid=rid, arrival_s=t, prompt_len=plen,
                                    output_len=olen, rtype=rt.name,
                                    alpha0=a0, alpha_decay=rt.alpha_decay))
                rid += 1
        reqs.sort(key=lambda r: r.arrival_s)
        return reqs


# ------------------------- canned workloads --------------------------------


def stationary(mix_name: str, rate: float, duration: float, seed: int = 0,
               draft_kind: str = "neural") -> WorkloadSpec:
    return WorkloadSpec(f"stationary-{mix_name}", [(duration, {mix_name: 1.0}, rate)],
                        seed=seed, draft_kind=draft_kind)


def mixed(rate: float, duration: float, seed: int = 0,
          draft_kind: str = "neural") -> WorkloadSpec:
    mix = {"rag": 0.3, "code": 0.2, "chat": 0.35, "reason": 0.15}
    return WorkloadSpec("mixed", [(duration, mix, rate)], seed=seed, draft_kind=draft_kind)


def step_perturbation(rate: float, warmup: float, post: float, seed: int = 0,
                      draft_kind: str = "neural") -> WorkloadSpec:
    """RQ1: a step change in the acceptance profile of arriving traffic.

    Phase 1 is high-acceptance (rag/code); at t=warmup the arriving mix flips to
    low-acceptance (chat/reason).  Both controllers must re-converge.
    """
    return WorkloadSpec(
        "step-perturbation",
        [(warmup, {"rag": 0.6, "code": 0.4}, rate),
         (post, {"chat": 0.6, "reason": 0.4}, rate)],
        seed=seed, draft_kind=draft_kind,
    )


def volatile(rate: float, duration: float, switch_period: float, seed: int = 0,
             draft_kind: str = "neural") -> WorkloadSpec:
    """RQ2 volatility axis: alternate high/low acceptance mixes every switch_period."""
    phases = []
    t = 0.0
    hi = {"rag": 0.6, "code": 0.4}
    lo = {"chat": 0.6, "reason": 0.4}
    i = 0
    while t < duration:
        d = min(switch_period, duration - t)
        phases.append((d, hi if i % 2 == 0 else lo, rate))
        t += d
        i += 1
    return WorkloadSpec(f"volatile-T{switch_period:g}", phases, seed=seed,
                        draft_kind=draft_kind)
