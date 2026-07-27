"""RQ1-RQ5 experiment drivers and the ablation matrices.

Every driver returns a tidy DataFrame (one row per run) so that the plotting
and LaTeX-table layers stay decoupled from the experiment definitions.
"""
from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import workload as W
from .controllers import (ADMIT_CONTROLLERS, COORDINATORS, SPEC_CONTROLLERS,
                          MIMOCoordinator, NaiveComposition)
from .metrics import end_metrics, is_oscillatory, stability_metrics
from .simulator import SimConfig, Simulator


# ==========================================================================
# Run plumbing
# ==========================================================================


def build_and_run(spec_name: str, admit_name: str, coord_name: str,
                  wl: W.WorkloadSpec, sim_cfg: SimConfig,
                  spec_kw: Optional[Dict] = None, admit_kw: Optional[Dict] = None,
                  coord_kw: Optional[Dict] = None, t_event: Optional[float] = None) -> Dict:
    spec_kw = dict(spec_kw or {})
    admit_kw = dict(admit_kw or {})
    coord_kw = dict(coord_kw or {})

    spec = SPEC_CONTROLLERS[spec_name](**spec_kw)
    admit = ADMIT_CONTROLLERS[admit_name](**admit_kw)
    coord = COORDINATORS[coord_name](**coord_kw) if coord_name != "none" else None

    hw = W.HARDWARE[sim_cfg.hardware]
    model = W.MODELS[sim_cfg.model]
    sim = Simulator(sim_cfg, hw, model, wl, spec, admit, coord)
    res = sim.run()

    row: Dict = {
        "spec": spec_name, "admit": admit_name, "coord": coord_name,
        "workload": wl.name, "model": sim_cfg.model, "hardware": sim_cfg.hardware,
        "seed": sim_cfg.seed, "draft_kind": wl.draft_kind,
        "spec_adaptive": spec.adaptive, "admit_adaptive": admit.adaptive,
        "spec_period": spec.period if spec.adaptive else -1,
        "admit_period": admit.period if admit.adaptive else -1,
        "spec_gain": spec.gain, "admit_gain": admit.gain,
    }
    row.update({f"kw_{k}": v for k, v in {**spec_kw, **admit_kw, **coord_kw}.items()
                if isinstance(v, (int, float, str))})
    row.update(end_metrics(res))
    row.update(stability_metrics(res, t_event=t_event))
    row["oscillatory"] = bool(is_oscillatory(row))
    return row


def _star(args):
    return build_and_run(**args)


def run_parallel(jobs: List[Dict], workers: Optional[int] = None) -> pd.DataFrame:
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if workers == 1 or len(jobs) == 1:
        return pd.DataFrame([_star(j) for j in jobs])
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_star, jobs))
    return pd.DataFrame(rows)


# ==========================================================================
# RQ1 - existence of interference (perturbation / system identification)
# ==========================================================================


def rq1_perturbation(model: str = "llama3.2-3b/1b", n_seeds: int = 24,
                     rate: float = 14.0, warmup: float = 120.0, post: float = 240.0,
                     spec_name: str = "ema", admit_name: str = "slack",
                     workers: Optional[int] = None) -> pd.DataFrame:
    """Four arms of the isolation ablation under a step disturbance.

    arm A: both static           (no adaptation at all)
    arm B: L_spec only           (static admission)
    arm C: L_admit only          (static gamma)
    arm D: both adaptive, naive  (the composed system under test)

    An interference claim requires D to show a signature (oscillation /
    settling time) absent from both B and C.
    """
    jobs = []
    for seed in range(n_seeds):
        wl = W.step_perturbation(rate=rate, warmup=warmup, post=post, seed=seed)
        cfg = SimConfig(model=model, seed=seed)
        arms = {
            "A_none": ("static", "static", dict(gamma=4), dict(max_batch=64)),
            "B_spec_only": (spec_name, "static", dict(period=4, gain=1.0), dict(max_batch=64)),
            "C_admit_only": ("static", admit_name, dict(gamma=4), dict(period=4, gain=1.0, max_batch_init=64)),
            "D_both_naive": (spec_name, admit_name, dict(period=4, gain=1.0),
                             dict(period=4, gain=1.0, max_batch_init=64)),
        }
        for arm, (s, a, skw, akw) in arms.items():
            jobs.append(dict(spec_name=s, admit_name=a, coord_name="naive", wl=wl,
                             sim_cfg=cfg, spec_kw=skw, admit_kw=akw,
                             t_event=warmup))
    df = run_parallel(jobs, workers)
    labels = ["A_none", "B_spec_only", "C_admit_only", "D_both_naive"]
    # arm labelling: executor.map preserves job order
    df["arm"] = np.tile(labels, n_seeds)[: len(df)]
    return df


# ==========================================================================
# RQ2 - conditions: gain x period-ratio x volatility stability map
# ==========================================================================


def rq2_sweep(model: str = "llama3.2-3b/1b", n_seeds: int = 5,
              spec_name: str = "ema", admit_name: str = "slack",
              gains=(0.25, 0.5, 1.0, 2.0, 4.0),
              period_ratios=(0.1, 0.25, 1.0, 4.0, 10.0),
              volatilities=(15.0, 45.0, 1e9),
              rate: float = 14.0, duration: float = 240.0,
              base_period: int = 4, workers: Optional[int] = None) -> pd.DataFrame:
    """Map the (gain, timescale-ratio, volatility) space into stable/oscillatory.

    period_ratio = admit_period / spec_period.  Ratio << 1 means admission is
    the fast loop; ratio >> 1 approximates timescale separation obtained for
    free.  volatility = mix switch period in seconds (1e9 == stationary).
    """
    jobs = []
    for g, pr, vol, seed in itertools.product(gains, period_ratios, volatilities, range(n_seeds)):
        wl = (W.mixed(rate=rate, duration=duration, seed=seed) if vol > 1e8
              else W.volatile(rate=rate, duration=duration, switch_period=vol, seed=seed))
        cfg = SimConfig(model=model, seed=seed)
        jobs.append(dict(
            spec_name=spec_name, admit_name=admit_name, coord_name="naive", wl=wl,
            sim_cfg=cfg,
            spec_kw=dict(period=base_period, gain=g),
            admit_kw=dict(period=max(1, int(round(base_period * pr))), gain=g,
                          max_batch_init=64),
        ))
    df = run_parallel(jobs, workers)
    grid = list(itertools.product(gains, period_ratios, volatilities, range(n_seeds)))
    df["gain"] = [x[0] for x in grid][: len(df)]
    df["period_ratio"] = [x[1] for x in grid][: len(df)]
    df["volatility_s"] = [x[2] for x in grid][: len(df)]
    return df


# ==========================================================================
# RQ3 - cost under realistic traces, vs the four counterfactual baselines
# ==========================================================================


def rq3_cost(model: str = "llama3.1-70b/8b", hardware: str = "H100-80Gx4",
             n_seeds: int = 10, rate: float = 6.0, duration: float = 300.0,
             spec_name: str = "ema", admit_name: str = "slack",
             workers: Optional[int] = None) -> pd.DataFrame:
    jobs = []
    for seed in range(n_seeds):
        wl = W.mixed(rate=rate, duration=duration, seed=seed)
        cfg = SimConfig(model=model, hardware=hardware, seed=seed, tpot_slo_s=0.060)
        arms = {
            "no_spec": ("static", "static", dict(gamma=0), dict(max_batch=64)),
            "static_spec_static_admit": ("static", "static", dict(gamma=4), dict(max_batch=64)),
            "adaptive_spec_only": (spec_name, "static", dict(period=4, gain=1.0), dict(max_batch=64)),
            "adaptive_admit_only": ("static", admit_name, dict(gamma=4),
                                    dict(period=4, gain=1.0, max_batch_init=64)),
            "both_naive": (spec_name, admit_name, dict(period=4, gain=1.0),
                           dict(period=4, gain=1.0, max_batch_init=64)),
        }
        for arm, (s, a, skw, akw) in arms.items():
            jobs.append(dict(spec_name=s, admit_name=a, coord_name="naive", wl=wl,
                             sim_cfg=cfg, spec_kw=skw, admit_kw=akw))
    df = run_parallel(jobs, workers)
    labels = ["no_spec", "static_spec_static_admit", "adaptive_spec_only",
              "adaptive_admit_only", "both_naive"]
    df["arm"] = np.tile(labels, n_seeds)[: len(df)]
    return df


# ==========================================================================
# RQ4 - coordination mechanisms
# ==========================================================================


def rq4_coordination(model: str = "llama3.1-70b/8b", hardware: str = "H100-80Gx4",
                     n_seeds: int = 10, rate: float = 6.0, duration: float = 300.0,
                     spec_name: str = "ema", admit_name: str = "slack",
                     warmup: float = 120.0, workers: Optional[int] = None) -> pd.DataFrame:
    """Compare naive composition against three coordination designs.

    Evaluated on both the perturbation trace (does the signature disappear?)
    and the mixed trace (is the end-metric benefit preserved?).
    """
    coords = {
        "naive": ("naive", {}),
        "timescale": ("timescale", dict(ratio=10, filter_beta=0.05)),
        "hysteresis": ("hysteresis", dict(cooldown_steps=20, deadband=0.10)),
        "mimo": ("mimo", dict(period=4, gamma_max=10, batch_max=256)),
    }
    jobs, tags = [], []
    for seed in range(n_seeds):
        for trace in ("mixed", "perturb"):
            wl = (W.mixed(rate=rate, duration=duration, seed=seed) if trace == "mixed"
                  else W.step_perturbation(rate=rate, warmup=warmup,
                                           post=duration - warmup, seed=seed))
            cfg = SimConfig(model=model, hardware=hardware, seed=seed, tpot_slo_s=0.060)
            for cname, (ckey, ckw) in coords.items():
                jobs.append(dict(spec_name=spec_name, admit_name=admit_name,
                                 coord_name=ckey, wl=wl, sim_cfg=cfg,
                                 spec_kw=dict(period=4, gain=1.0),
                                 admit_kw=dict(period=4, gain=1.0, max_batch_init=64),
                                 coord_kw=ckw,
                                 t_event=warmup if trace == "perturb" else None))
                tags.append((cname, trace))
    df = run_parallel(jobs, workers)
    df["coord_name"] = [x[0] for x in tags][: len(df)]
    df["trace"] = [x[1] for x in tags][: len(df)]
    return df


# ==========================================================================
# RQ5 - generality across draft strategies, architectures, and controllers
# ==========================================================================


def rq5_generality(n_seeds: int = 6, rate: float = 6.0, duration: float = 240.0,
                   workers: Optional[int] = None) -> pd.DataFrame:
    """Model/draft-strategy sweep, naive composition vs best coordination."""
    configs = [
        ("llama3.2-3b/1b", "H100-80G", "neural", 12.0),
        ("llama3.1-70b/8b", "H100-80Gx4", "neural", 6.0),
        ("qwen2.5-32b/1.5b", "H100-80Gx4", "neural", 8.0),
        ("moe-8x7b/1b", "H100-80Gx4", "neural", 8.0),
        ("llama3.1-8b/eagle", "H100-80G", "neural", 10.0),
        ("llama3.1-8b/ngram", "H100-80G", "ngram", 10.0),
    ]
    jobs, tags = [], []
    for model, hw, dk, r in configs:
        for seed in range(n_seeds):
            wl = W.mixed(rate=r, duration=duration, seed=seed, draft_kind=dk)
            cfg = SimConfig(model=model, hardware=hw, seed=seed, tpot_slo_s=0.060)
            for coord, ckw in (("naive", {}), ("mimo", dict(period=4))):
                jobs.append(dict(spec_name="ema", admit_name="slack", coord_name=coord,
                                 wl=wl, sim_cfg=cfg, spec_kw=dict(period=4, gain=1.0),
                                 admit_kw=dict(period=4, gain=1.0, max_batch_init=64),
                                 coord_kw=ckw))
                tags.append((model, dk, coord))
    df = run_parallel(jobs, workers)
    df["cfg_model"] = [x[0] for x in tags][: len(df)]
    df["cfg_draft"] = [x[1] for x in tags][: len(df)]
    df["cfg_coord"] = [x[2] for x in tags][: len(df)]
    return df


# ==========================================================================
# Ablation B: controller-design ablation (SOTA L_spec control laws)
# ==========================================================================


def ablation_controller_design(model: str = "llama3.1-70b/8b", hardware: str = "H100-80Gx4",
                               n_seeds: int = 8, rate: float = 6.0, duration: float = 300.0,
                               warmup: float = 120.0, workers: Optional[int] = None) -> pd.DataFrame:
    """Hold L_admit fixed; swap L_spec across published control laws.

    If interference appears for every SOTA speculation controller, the finding
    is a property of the composition, not of any one controller's design.
    """
    spec_variants = ["ema", "bandit", "entropy", "dsde"]
    admit_variants = ["slack", "queue"]
    jobs, tags = [], []
    for seed in range(n_seeds):
        wl = W.step_perturbation(rate=rate, warmup=warmup, post=duration - warmup, seed=seed)
        cfg = SimConfig(model=model, hardware=hardware, seed=seed, tpot_slo_s=0.060)
        for s in spec_variants:
            for a in admit_variants:
                # composed
                jobs.append(dict(spec_name=s, admit_name=a, coord_name="naive", wl=wl,
                                 sim_cfg=cfg, spec_kw=dict(period=4, gain=1.0),
                                 admit_kw=dict(period=4, gain=1.0, max_batch_init=64),
                                 t_event=warmup))
                tags.append((s, a, "composed"))
                # isolated L_spec (static admission) - the control condition
                jobs.append(dict(spec_name=s, admit_name="static", coord_name="naive", wl=wl,
                                 sim_cfg=cfg, spec_kw=dict(period=4, gain=1.0),
                                 admit_kw=dict(max_batch=64), t_event=warmup))
                tags.append((s, a, "isolated"))
    df = run_parallel(jobs, workers)
    df["spec_variant"] = [x[0] for x in tags][: len(df)]
    df["admit_variant"] = [x[1] for x in tags][: len(df)]
    df["condition"] = [x[2] for x in tags][: len(df)]
    return df


# ==========================================================================
# Ablation C: loop-count
# ==========================================================================


def ablation_loop_count(model: str = "llama3.1-70b/8b", hardware: str = "H100-80Gx4",
                        n_seeds: int = 8, rate: float = 6.0, duration: float = 300.0,
                        workers: Optional[int] = None) -> pd.DataFrame:
    """1 loop -> 2 loops -> 2 loops under KV pressure (the implicit third loop).

    The KV/preemption loop is exercised by shrinking the KV budget, which makes
    the simulator's speculative-reservation shrink path active.
    """
    jobs, tags = [], []
    for seed in range(n_seeds):
        wl = W.mixed(rate=rate, duration=duration, seed=seed)
        for nloops, kvfrac in ((1, 0.85), (2, 0.85), (3, 0.35)):
            cfg = SimConfig(model=model, hardware=hardware, seed=seed,
                            tpot_slo_s=0.060, kv_capacity_frac=kvfrac)
            if nloops == 1:
                s, a = "ema", "static"
                skw, akw = dict(period=4, gain=1.0), dict(max_batch=64)
            else:
                s, a = "ema", "slack"
                skw, akw = dict(period=4, gain=1.0), dict(period=4, gain=1.0, max_batch_init=64)
            jobs.append(dict(spec_name=s, admit_name=a, coord_name="naive", wl=wl,
                             sim_cfg=cfg, spec_kw=skw, admit_kw=akw))
            tags.append(nloops)
    df = run_parallel(jobs, workers)
    df["n_loops"] = tags[: len(df)]
    return df


# ==========================================================================
# Ablation E: SLO tightness x workload volatility
# ==========================================================================


def ablation_slo_workload(model: str = "llama3.1-70b/8b", hardware: str = "H100-80Gx4",
                          n_seeds: int = 6, rate: float = 6.0, duration: float = 240.0,
                          slos=(0.030, 0.060, 0.120), vols=(15.0, 45.0, 1e9),
                          workers: Optional[int] = None) -> pd.DataFrame:
    jobs, tags = [], []
    for slo, vol, seed in itertools.product(slos, vols, range(n_seeds)):
        wl = (W.mixed(rate=rate, duration=duration, seed=seed) if vol > 1e8
              else W.volatile(rate=rate, duration=duration, switch_period=vol, seed=seed))
        cfg = SimConfig(model=model, hardware=hardware, seed=seed, tpot_slo_s=slo)
        jobs.append(dict(spec_name="ema", admit_name="slack", coord_name="naive", wl=wl,
                         sim_cfg=cfg, spec_kw=dict(period=4, gain=1.0),
                         admit_kw=dict(period=4, gain=1.0, max_batch_init=64)))
        tags.append((slo, vol))
    df = run_parallel(jobs, workers)
    df["tpot_slo"] = [x[0] for x in tags][: len(df)]
    df["volatility_s"] = [x[1] for x in tags][: len(df)]
    return df


# ==========================================================================
# Pilot: the cheap go/no-go test described in the plan
# ==========================================================================


def pilot(model: str = "llama3.2-3b/1b", n_seeds: int = 8,
          workers: Optional[int] = None) -> pd.DataFrame:
    """One-hour version of RQ1. If arm D shows no signature relative to B and C,
    the premise is wrong and nothing else needs to be built."""
    return rq1_perturbation(model=model, n_seeds=n_seeds, rate=14.0,
                            warmup=60.0, post=120.0, workers=workers)


# ==========================================================================
# Mechanism isolation: acceptance heterogeneity as the oscillation channel
# ==========================================================================


def _install_spread_types():
    """Two request types identical in every respect except acceptance spread."""
    base = W.REQUEST_TYPES["chat"]
    for name, sigma in (("homog", 0.001), ("hetero", 0.18)):
        W.REQUEST_TYPES[name] = W.RequestType(
            name, prompt_mu=base.prompt_mu, prompt_sigma=base.prompt_sigma,
            output_mu=base.output_mu, output_sigma=base.output_sigma,
            alpha_mu=0.70, alpha_sigma=sigma, alpha_ngram_mu=0.70, alpha_decay=0.0)


def mechanism_alpha_spread(model: str = "llama3.2-3b/1b", n_seeds: int = 6,
                           rate: float = 24.0, duration: float = 90.0,
                           spec_variants=("ema", "entropy", "bandit", "dsde"),
                           workers: Optional[int] = None) -> pd.DataFrame:
    """Does batch-aggregate acceptance sensing self-oscillate?

    Single control loop only (admission held static), so nothing here can be
    attributed to loop composition.  ``homog`` and ``hetero`` share the same
    mean acceptance, prompt/output distributions, and arrival process; they
    differ only in the per-request spread of alpha.  If the oscillation appears
    only under ``hetero``, the feedback channel is batch composition: raising
    gamma drains high-alpha requests faster, which lowers the batch-mean alpha
    the controller is sensing, which lowers gamma, which lets them accumulate
    again.
    """
    _install_spread_types()
    jobs, tags = [], []
    for spread in ("homog", "hetero"):
        for spec in spec_variants:
            for seed in range(n_seeds):
                wl = W.stationary(spread, rate=rate, duration=duration, seed=seed)
                cfg = SimConfig(model=model, seed=seed, tpot_slo_s=0.020,
                                max_sim_s=duration + 20.0)
                jobs.append(dict(spec_name=spec, admit_name="static", coord_name="naive",
                                 wl=wl, sim_cfg=cfg, spec_kw=dict(period=4, gain=0.5),
                                 admit_kw=dict(max_batch=64)))
                tags.append((spread, spec))
    df = run_parallel(jobs, workers)
    df["alpha_spread"] = [x[0] for x in tags][: len(df)]
    df["spec_variant"] = [x[1] for x in tags][: len(df)]
    return df


def regime_sweep(model: str = "llama3.2-3b/1b", n_seeds: int = 4,
                 rates=(14.0, 24.0, 48.0), slos=(0.020, 0.050),
                 warmup: float = 30.0, post: float = 60.0,
                 workers: Optional[int] = None) -> pd.DataFrame:
    """Load/SLO regime sweep for the isolation arms.

    Run this BEFORE committing to the two-loop framing: it establishes whether
    a regime exists in which naive composition actually costs end-metrics.
    """
    jobs, tags = [], []
    for rate in rates:
        for slo in slos:
            for seed in range(n_seeds):
                wl = W.step_perturbation(rate=rate, warmup=warmup, post=post, seed=seed)
                cfg = SimConfig(model=model, seed=seed, tpot_slo_s=slo,
                                max_sim_s=warmup + post + 20.0)
                arms = {
                    "B_spec": ("ema", "static", dict(period=4, gain=0.5), dict(max_batch=64)),
                    "C_admit": ("static", "slack", dict(gamma=4),
                                dict(period=4, gain=0.5, max_batch_init=64)),
                    "D_both": ("ema", "slack", dict(period=4, gain=0.5),
                               dict(period=4, gain=0.5, max_batch_init=64)),
                }
                for arm, (s, a, skw, akw) in arms.items():
                    jobs.append(dict(spec_name=s, admit_name=a, coord_name="naive", wl=wl,
                                     sim_cfg=cfg, spec_kw=skw, admit_kw=akw, t_event=warmup))
                    tags.append((rate, slo, arm))
    df = run_parallel(jobs, workers)
    df["rate"] = [x[0] for x in tags][: len(df)]
    df["slo"] = [x[1] for x in tags][: len(df)]
    df["arm"] = [x[2] for x in tags][: len(df)]
    return df
