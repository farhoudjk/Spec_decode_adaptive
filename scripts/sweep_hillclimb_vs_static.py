"""Axis-4 roofline follow-up: does HillClimbSpec (controllers.HillClimbSpec)
beat every fixed static k, at every batch size?

The B x k grid in axis4_roofline_moe_fit established that (a) speculation has
real leverage on Mixtral-8x7B-FP8+ngram (21-34% ITL improvement over k=0 at
every B), and (b) the ITL-minimizing k is not fixed -- it falls as batch grows
(measured optimum: B=8->4, B=16->6 (broad), B=24->3, B=32->2). No single
static k is optimal across that whole range, which is exactly the condition
under which an adaptive controller has room to win.

HillClimbSpec closes the loop on MEASURED ITL (obs.tpot_ema) directly, with
the search window bounded by live batch occupancy (obs.num_running) rather
than a fixed range -- see the class docstring in controllers.py for why ITL
and not acceptance (acceptance falls monotonically across the whole k range in
the fit sweep, so it can't localize an interior optimum) or a static k*(B)
lookup (doesn't self-correct if live conditions diverge from the sweep this
was fit on).

This sweep runs, at each of B in {8, 16, 24, 32}:
  hillclimb        -- HillClimbSpec, batch cap fixed at B (static admit)
  static-k1..k8     -- StaticSpec at every k tested in the fit sweep

so "hillclimb beats every fixed k at its own batch size" is a real comparison,
not assumed. Batch cap is STATIC (admit: static) at each B throughout --
this isolates the spec-controller comparison from any admission-controller
effect, matching the fit sweep's isolation of (B, k).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from specloop_rt.analysis import summarize_run

STATIC_KS = [1, 2, 3, 4, 5, 6, 8]


def build_arm(arm: str, B: int) -> dict:
    admit_kw = {"max_num_seqs": B}
    if arm == "hillclimb":
        return {"controller": {"spec": "hillclimb", "admit": "static",
                               "coordination": "naive",
                               "spec_kw": {"settle_steps": 40, "deadband_frac": 0.03,
                                           "gamma_init": 4},
                               "admit_kw": admit_kw}}
    if arm.startswith("static-k"):
        k = int(arm[len("static-k"):])
        return {"controller": {"spec": "static", "admit": "static",
                               "coordination": "naive",
                               "spec_kw": {"gamma": k},
                               "admit_kw": admit_kw}}
    raise KeyError(arm)


def run_cell(base_cfg, arm, B, rtype, rate, duration, seed, out_dir):
    cfg = copy.deepcopy(base_cfg)
    spec = build_arm(arm, B)
    cfg["runtime"]["max_num_seqs_init"] = B
    # NOTE: cfg["runtime"]["gamma_init"] is what _build_engine (replay.py)
    # reads to construct speculative_config -- spec["controller"]["spec_kw"]
    # ["gamma_init"] is a SEPARATE field the engine-builder never reads. Before
    # the live_gamma_patch (specloop_rt/vllm_patch/live_gamma_patch.py) fixed
    # the underlying proposer freeze, every cell in this sweep silently ran at
    # whatever gamma_init the base config had (4), regardless of the arm's
    # configured k -- see AXIS5_ROOFLINE_MOE.md#4. Setting it here for both
    # static-k arms (their real k) and hillclimb (its starting gamma) keeps
    # engine-construction k consistent with what each arm claims, on top of
    # (not instead of) the live-actuation fix -- static's k should never move
    # after construction, and now genuinely won't.
    static_k = spec["controller"]["spec_kw"].get("gamma")
    hillclimb_init = spec["controller"]["spec_kw"].get("gamma_init")
    cfg["runtime"]["gamma_init"] = (static_k if static_k is not None
                                    else hillclimb_init if hillclimb_init is not None
                                    else cfg["runtime"].get("gamma_init", 4))
    cfg["controller"] = copy.deepcopy(spec["controller"])
    cfg_path = os.path.join(out_dir, f"cfg_{arm}_B{B}_{rtype}_r{rate}_s{seed}.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    run_dir = os.path.join(out_dir, f"{arm}_B{B}_{rtype}_r{rate}_s{seed}")
    cmd = [sys.executable, "-m", "specloop_rt.replay", "--config", cfg_path,
           "--trace", "homogeneous", "--rtype", rtype, "--rate", str(rate),
           "--duration", str(duration), "--seed", str(seed),
           "--real-corpus", "--out", run_dir]
    print(f">> arm={arm} B={B} rtype={rtype} rate={rate} seed={seed}", flush=True)
    subprocess.run(cmd, check=True)
    return summarize_run(run_dir, tpot_slo=base_cfg["runtime"]["tpot_slo_s"],
                         ttft_slo=base_cfg["runtime"]["ttft_slo_s"])


def main(argv=None):
    p = argparse.ArgumentParser("hillclimb vs static-k, all batch sizes")
    p.add_argument("--config", required=True)
    p.add_argument("--rtype", default="code")
    p.add_argument("--rate", type=float, default=8.0)
    p.add_argument("--Bs", type=int, nargs="+", default=[8, 16, 24, 32])
    p.add_argument("--ks", type=int, nargs="+", default=STATIC_KS)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--out", default="results_gpu_sweep/axis4_hillclimb_vs_static")
    a = p.parse_args(argv)

    arms = ["hillclimb"] + [f"static-k{k}" for k in a.ks]

    os.makedirs(a.out, exist_ok=True)
    grid_path = os.path.join(a.out, "grid.json")
    grid = json.load(open(grid_path)) if os.path.exists(grid_path) else {}

    with open(a.config) as f:
        base_cfg = yaml.safe_load(f)

    for B in a.Bs:
        for arm in arms:
            for seed in a.seeds:
                key = f"{arm}_B{B}_{a.rtype}_r{a.rate}_s{seed}"
                if key in grid and "error" not in grid[key]:
                    print(f".. skip {key}", flush=True)
                    continue
                try:
                    m = run_cell(base_cfg, arm, B, a.rtype, a.rate, a.duration,
                                 seed, a.out)
                    grid[key] = {"arm": arm, "B": B, "rtype": a.rtype,
                                 "rate": a.rate, "seed": seed, **m}
                except subprocess.CalledProcessError as e:
                    print(f"!! {key} FAILED: {e}", flush=True)
                    grid[key] = {"arm": arm, "B": B, "rtype": a.rtype,
                                 "rate": a.rate, "seed": seed, "error": str(e)}
                with open(grid_path, "w") as f:
                    json.dump(grid, f, indent=2)

    # ---- per-B comparison: hillclimb vs best/worst static-k ----
    print("\n=== hillclimb vs static-k, by batch size (tpot_p50, s) ===")
    for B in a.Bs:
        rows = {v["arm"]: v["tpot_p50"] for v in grid.values()
                if v.get("B") == B and "error" not in v and "tpot_p50" in v}
        if "hillclimb" not in rows:
            continue
        hc = rows.pop("hillclimb")
        if not rows:
            continue
        best_static_arm = min(rows, key=rows.get)
        best_static = rows[best_static_arm]
        worst_static = max(rows.values())
        print(f"B={B:>3}  hillclimb={hc:.4f}  best_static={best_static:.4f} ({best_static_arm})  "
              f"worst_static={worst_static:.4f}  "
              f"hillclimb_vs_best={100*(hc-best_static)/best_static:+.1f}%")


if __name__ == "__main__":
    main()
