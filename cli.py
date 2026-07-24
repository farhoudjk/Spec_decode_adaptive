"""Command-line driver.

    python -m specloop.cli pilot          # go/no-go, ~minutes
    python -m specloop.cli rq1 --seeds 24
    python -m specloop.cli rq2 --seeds 5
    python -m specloop.cli rq3 rq4 rq5
    python -m specloop.cli abl-controller abl-loops abl-slo
    python -m specloop.cli all
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict

import numpy as np
import pandas as pd

from . import experiments as E
from . import report as R
from . import workload as W
from .controllers import ADMIT_CONTROLLERS, COORDINATORS, SPEC_CONTROLLERS
from .simulator import SimConfig, Simulator

OUT = os.environ.get("SPECLOOP_OUT", "results")


def _save(df: pd.DataFrame, name: str) -> str:
    os.makedirs(f"{OUT}/csv", exist_ok=True)
    p = f"{OUT}/csv/{name}.csv"
    df.to_csv(p, index=False)
    return p


def _fig(name: str) -> str:
    os.makedirs(f"{OUT}/fig", exist_ok=True)
    return f"{OUT}/fig/{name}"


def _tab(name: str) -> str:
    os.makedirs(f"{OUT}/tab", exist_ok=True)
    return f"{OUT}/tab/{name}"


def _exemplar_traces(model: str, seed: int, warmup: float, post: float, rate: float):
    """Single-seed traces for the RQ1 time-series figure."""
    traces = {}
    arms = {
        "static/static": ("static", "static", dict(gamma=4), dict(max_batch=64)),
        r"$L_{spec}$ only": ("ema", "static", dict(period=4, gain=1.0), dict(max_batch=64)),
        r"$L_{admit}$ only": ("static", "slack", dict(gamma=4), dict(period=4, gain=1.0, max_batch_init=64)),
        "both (naive)": ("ema", "slack", dict(period=4, gain=1.0), dict(period=4, gain=1.0, max_batch_init=64)),
    }
    for label, (s, a, skw, akw) in arms.items():
        wl = W.step_perturbation(rate=rate, warmup=warmup, post=post, seed=seed)
        cfg = SimConfig(model=model, seed=seed)
        sim = Simulator(cfg, W.HARDWARE[cfg.hardware], W.MODELS[model], wl,
                        SPEC_CONTROLLERS[s](**skw), ADMIT_CONTROLLERS[a](**akw),
                        COORDINATORS["naive"]())
        traces[label] = sim.run().to_frame()
    return traces


def cmd_pilot(a):
    t0 = time.time()
    df = E.pilot(model=a.model, n_seeds=a.seeds, workers=a.workers)
    _save(df, "pilot")
    g = df.groupby("arm")[["gamma_spectral_concentration", "gamma_cv",
                           "gamma_settling_s", "slo_attainment"]].mean()
    print(g.to_string())
    d = g.loc["D_both_naive", "gamma_spectral_concentration"]
    b = max(g.loc["B_spec_only", "gamma_spectral_concentration"],
            g.loc["C_admit_only", "gamma_spectral_concentration"])
    verdict = "SIGNATURE PRESENT" if d > 2 * max(b, 1e-6) else "NO SIGNATURE - premise not supported"
    print(f"\npilot verdict: {verdict}  (composed={d:.4f} vs best-isolated={b:.4f})")
    print(f"[{time.time()-t0:.1f}s]")


def cmd_rq1(a):
    df = E.rq1_perturbation(model=a.model, n_seeds=a.seeds, workers=a.workers)
    _save(df, "rq1_perturbation")
    R.fig_rq1_bars(df, _fig("rq1_signatures.pdf"))
    traces = _exemplar_traces(a.model, seed=0, warmup=120.0, post=240.0, rate=14.0)
    R.fig_rq1_timeseries(traces, t_event=120.0, path=_fig("rq1_timeseries.pdf"))
    R.table_e2e(df, _tab("rq1_arms.tex"), group="arm",
                order=["A_none", "B_spec_only", "C_admit_only", "D_both_naive"],
                caption="Isolation ablation under a step disturbance in the arriving acceptance profile.",
                label="tab:rq1")
    print(df.groupby("arm")[["gamma_spectral_concentration", "gamma_cv",
                             "gamma_settling_s", "slo_attainment", "goodput_tok_s"]]
          .mean().to_string())


def cmd_rq2(a):
    df = E.rq2_sweep(model=a.model, n_seeds=a.seeds, workers=a.workers)
    _save(df, "rq2_sweep")
    R.fig_rq2_boundary(df, _fig("rq2_boundary.pdf"))
    R.table_stability_map(df, _tab("rq2_stability_map.tex"))
    print(df.groupby(["period_ratio", "gain"])["oscillatory"].mean().unstack().to_string())


def cmd_rq3(a):
    df = E.rq3_cost(n_seeds=a.seeds, workers=a.workers)
    _save(df, "rq3_cost")
    R.fig_rq3_cost(df, _fig("rq3_cost.pdf"))
    R.table_e2e(df, _tab("rq3_cost.tex"), group="arm",
                order=["no_spec", "static_spec_static_admit", "adaptive_spec_only",
                       "adaptive_admit_only", "both_naive"],
                caption="End-to-end cost of naive loop composition on a mixed production-style trace.",
                label="tab:rq3")
    print(df.groupby("arm")[["slo_attainment", "tpot_p99", "goodput_tok_s",
                             "gpu_s_per_ktok"]].mean().to_string())


def cmd_rq4(a):
    df = E.rq4_coordination(n_seeds=a.seeds, workers=a.workers)
    _save(df, "rq4_coordination")
    R.fig_rq4_coordination(df[df.trace == "mixed"], _fig("rq4_coordination.pdf"))
    R.table_e2e(df[df.trace == "mixed"], _tab("rq4_coordination.tex"), group="coord_name",
                order=["naive", "timescale", "hysteresis", "mimo"],
                caption="Coordination mechanisms on the mixed trace.", label="tab:rq4")
    print(df.groupby(["trace", "coord_name"])[
        ["gamma_spectral_concentration", "gamma_settling_s", "slo_attainment",
         "goodput_tok_s"]].mean().to_string())


def cmd_rq5(a):
    df = E.rq5_generality(n_seeds=a.seeds, workers=a.workers)
    _save(df, "rq5_generality")
    R.table_e2e(df[df.cfg_coord == "naive"], _tab("rq5_generality.tex"), group="cfg_model",
                order=sorted(df.cfg_model.unique()),
                caption="Generality of the interference signature across architectures and draft strategies.",
                label="tab:rq5")
    print(df.groupby(["cfg_model", "cfg_coord"])[
        ["gamma_spectral_concentration", "slo_attainment", "goodput_tok_s"]]
        .mean().to_string())


def cmd_abl_controller(a):
    df = E.ablation_controller_design(n_seeds=a.seeds, workers=a.workers)
    _save(df, "ablation_controller_design")
    R.table_controller_ablation(df, _tab("ablation_controller.tex"))
    print(df.groupby(["spec_variant", "admit_variant", "condition"])[
        "gamma_spectral_concentration"].mean().unstack().to_string())


def cmd_abl_loops(a):
    df = E.ablation_loop_count(n_seeds=a.seeds, workers=a.workers)
    _save(df, "ablation_loop_count")
    print(df.groupby("n_loops")[["gamma_spectral_concentration", "gamma_cv",
                                 "slo_attainment", "preemptions"]].mean().to_string())


def cmd_abl_slo(a):
    df = E.ablation_slo_workload(n_seeds=a.seeds, workers=a.workers)
    _save(df, "ablation_slo_workload")
    print(df.groupby(["tpot_slo", "volatility_s"])["oscillatory"].mean().unstack().to_string())


def cmd_mechanism(a):
    df = E.mechanism_alpha_spread(model=a.model, n_seeds=a.seeds, workers=a.workers)
    _save(df, "mechanism_alpha_spread")
    g = df.groupby(["spec_variant", "alpha_spread"])[
        ["gamma_spectral_concentration", "gamma_cv", "mean_gamma",
         "goodput_tok_s", "tpot_p99"]].mean()
    print(g.to_string())
    R.table_e2e(df[df.alpha_spread == "hetero"], _tab("mechanism.tex"),
                group="spec_variant", order=sorted(df.spec_variant.unique()),
                caption="Speculation controllers under heterogeneous per-request acceptance, single loop.",
                label="tab:mechanism")


def cmd_regime(a):
    df = E.regime_sweep(model=a.model, n_seeds=a.seeds, workers=a.workers)
    _save(df, "regime_sweep")
    print(df.groupby(["rate", "slo", "arm"])[
        ["gamma_spectral_concentration", "tpot_p99", "slo_attainment",
         "goodput_tok_s"]].mean().to_string())


COMMANDS = {
    "pilot": cmd_pilot, "mechanism": cmd_mechanism, "regime": cmd_regime, "rq1": cmd_rq1, "rq2": cmd_rq2, "rq3": cmd_rq3,
    "rq4": cmd_rq4, "rq5": cmd_rq5, "abl-controller": cmd_abl_controller,
    "abl-loops": cmd_abl_loops, "abl-slo": cmd_abl_slo,
}


def main(argv=None):
    p = argparse.ArgumentParser(prog="specloop")
    p.add_argument("commands", nargs="+", choices=list(COMMANDS) + ["all"])
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--model", default="llama3.2-3b/1b")
    p.add_argument("--workers", type=int, default=None)
    a = p.parse_args(argv)
    cmds = list(COMMANDS) if "all" in a.commands else a.commands
    for c in cmds:
        print(f"\n{'='*72}\n{c}\n{'='*72}")
        COMMANDS[c](a)


if __name__ == "__main__":
    main()
