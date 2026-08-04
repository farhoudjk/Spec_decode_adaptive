"""Axis-4 roofline follow-up: direct B x k grid on Mixtral-8x7B (MoE, FP8) +
EAGLE, to test AXIS4_ROOFLINE.md's central claim -- that speculation has
leverage (ITL falls as k rises) while B < B*, and loses leverage once B >= B*.

Unlike sweep_predictive.py (which sweeps admission-controller ARMS at fixed
rate/gamma to test whether a controller reacts), this sweep holds the
controller STATIC (fixed batch cap, fixed gamma) and sweeps (B, k) directly,
so the measured ITL(B,k) surface can be compared against the closed-form
prediction from the theory doc:

    T_step(B,k) = max(T_mem(B), T_comp(B,k)) + k*t_d
    ITL(k)      = T_step / E[tokens/step]

For Mixtral-8x7B-FP8 on a single A100-80GB (see configs/a100_80gb_mixtral_
eagle.yaml header for the derivation), B* ~ 20-25 depending on k -- so this
sweep straddles that with B in {8, 16, 24, 32, 48} and k in {1, 2, 4, 6, 8}.
The prediction under test (theory sec 4.2, costed drafter t_d>0): for B <
B*, ITL(k) should have an interior minimum tracking measured acceptance
alpha; for B >= B*, ITL(k) should be roughly flat or increasing in k (k
becomes inert / actively harmful because verify cost no longer overlaps
idle compute).

Each cell pins gamma_init=k and max_num_seqs_init=B via the static/static
controller (no adaptive movement mid-run), so tpot_p50 in the resulting
grid.json IS the ITL(B,k) sample point.
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

DEFAULT_B = [8, 16, 24, 32, 48]
DEFAULT_K = [1, 2, 4, 6, 8]


def run_cell(base_cfg, B, k, rtype, rate, duration, seed, out_dir):
    cfg = copy.deepcopy(base_cfg)
    cfg["runtime"]["max_num_seqs_init"] = B
    cfg["runtime"]["gamma_init"] = k
    cfg["controller"] = {
        "spec": "static", "admit": "static", "coordination": "naive",
        "spec_kw": {"gamma": k},
        "admit_kw": {"max_num_seqs": B},
        "coord_kw": {},
    }
    cfg_path = os.path.join(out_dir, f"cfg_B{B}_k{k}_{rtype}_r{rate}_s{seed}.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    run_dir = os.path.join(out_dir, f"B{B}_k{k}_{rtype}_r{rate}_s{seed}")
    cmd = [sys.executable, "-m", "specloop_rt.replay", "--config", cfg_path,
           "--trace", "homogeneous", "--rtype", rtype, "--rate", str(rate),
           "--duration", str(duration), "--seed", str(seed),
           "--real-corpus", "--out", run_dir]
    print(f">> B={B} k={k} rtype={rtype} rate={rate} seed={seed}", flush=True)
    subprocess.run(cmd, check=True)
    return summarize_run(run_dir, tpot_slo=base_cfg["runtime"]["tpot_slo_s"],
                         ttft_slo=base_cfg["runtime"]["ttft_slo_s"])


def main(argv=None):
    p = argparse.ArgumentParser("axis-4 roofline MoE B*/k* sweep")
    p.add_argument("--config", required=True)
    p.add_argument("--rtype", default="code")
    p.add_argument("--rate", type=float, default=8.0,
                   help="arrival rate must be high enough that the batch cap "
                        "B actually binds -- otherwise the run never reaches "
                        "B concurrent sequences and the cell doesn't test "
                        "what it claims to. Verify against mean_running in "
                        "the resulting steps.jsonl.")
    p.add_argument("--Bs", type=int, nargs="+", default=DEFAULT_B)
    p.add_argument("--ks", type=int, nargs="+", default=DEFAULT_K)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--duration", type=float, default=180.0)
    p.add_argument("--out", default="results_gpu_sweep/axis4_roofline_moe")
    a = p.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    grid_path = os.path.join(a.out, "grid.json")
    grid = json.load(open(grid_path)) if os.path.exists(grid_path) else {}

    with open(a.config) as f:
        base_cfg = yaml.safe_load(f)

    for B in a.Bs:
        for k in a.ks:
            for seed in a.seeds:
                key = f"B{B}_k{k}_{a.rtype}_r{a.rate}_s{seed}"
                if key in grid and "error" not in grid[key]:
                    print(f".. skip {key}", flush=True)
                    continue
                try:
                    m = run_cell(base_cfg, B, k, a.rtype, a.rate, a.duration,
                                 seed, a.out)
                    grid[key] = {"B": B, "k": k, "rtype": a.rtype,
                                 "rate": a.rate, "seed": seed, **m}
                except subprocess.CalledProcessError as e:
                    print(f"!! {key} FAILED: {e}", flush=True)
                    grid[key] = {"B": B, "k": k, "rtype": a.rtype,
                                 "rate": a.rate, "seed": seed, "error": str(e)}
                with open(grid_path, "w") as f:
                    json.dump(grid, f, indent=2)

    # ---- seed-averaged ITL(B,k) surface ----
    import statistics as st
    print("\n=== ITL (tpot_p50, s) by B x k, mean+-sd over seeds ===")
    header = "B\\k".ljust(6) + "".join(str(k).rjust(16) for k in a.ks)
    print(header)
    for B in a.Bs:
        row = str(B).ljust(6)
        for k in a.ks:
            vals = [v["tpot_p50"] for v in grid.values()
                    if v.get("B") == B and v.get("k") == k and "error" not in v
                    and "tpot_p50" in v]
            if not vals:
                row += "n/a".rjust(16)
                continue
            mu = sum(vals) / len(vals)
            sd = st.stdev(vals) if len(vals) > 1 else 0.0
            row += f"{mu:.4f}+-{sd:.4f}".rjust(16)
        print(row)

    print("\n=== mean acceptance alpha by B x k, mean over seeds ===")
    print(header)
    for B in a.Bs:
        row = str(B).ljust(6)
        for k in a.ks:
            vals = [v["mean_accept_rate"] for v in grid.values()
                    if v.get("B") == B and v.get("k") == k and "error" not in v
                    and "mean_accept_rate" in v and v["mean_accept_rate"] == v["mean_accept_rate"]]
            if not vals:
                row += "n/a".rjust(16)
                continue
            mu = sum(vals) / len(vals)
            row += f"{mu:.3f}".rjust(16)
        print(row)


if __name__ == "__main__":
    main()
