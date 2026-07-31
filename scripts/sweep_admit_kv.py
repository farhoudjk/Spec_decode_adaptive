"""Axis-3 sweep: admission-primary, gamma-secondary controller vs. bracketing arms.

Companion to sweep_boundedness.py (axis 1: locates the decode-bound regime)
and sweep_workload_gamma.py (axis 2: gamma-spread stays noise-level at both
admission caps). Axis 1 and Axis 2 together are the case FOR this design: the
cap moved SLO/TTFT sharply while gamma moved goodput/acceptance by <10% of
the mean everywhere. This sweep is the first one that actually runs the
adaptive controllers instead of static gamma throughout -- it tests whether
KVAwareAdmit + GatedSpec (specloop_rt/controllers.py) converts that
observational finding into a real end-metric win, and isolates which piece of
the controller (cap loop vs. KV ceiling vs. gated gamma) is doing the work.

Five bracketing arms, run at every (rtype, rate) cell so no arm's apparent win
is an artifact of one workload/rate:

  static-cap-lo      static admit at a deliberately tight cap, static gamma.
                     The floor any adaptive cap must beat -- without this arm,
                     "adaptive beats a generous static cap" is not evidence of
                     anything, since a generous cap is not the failure mode
                     TTFT-slack admission targets (same logic as static-low-k
                     in the Axis-2 companion doc: any adaptive law needs a
                     losing constant, not just the current default, as its
                     lower bracket).
  static-cap-hi      static admit at the generous cap Axis-1/Axis-2 already
                     used, static gamma. The baseline these axes were run at.
  adaptive-cap-only  TTFTSlackAdmit (no KV ceiling) + gamma pinned at
                     gamma_floor. Isolates whether the TTFT-slack cap loop by
                     itself captures the whole benefit, before crediting KV.
  kv-aware-cap-only  KVAwareAdmit (TTFT-slack + KV ceiling) + gamma pinned at
                     gamma_floor. Isolates the KV ceiling's incremental
                     contribution over the TTFT-only cap loop -- the
                     e2e-p95/ITL claim lives or dies on the gap between this
                     arm and adaptive-cap-only.
  full               KVAwareAdmit + GatedSpec. The full proposed controller.

Usage:
    python3 scripts/sweep_admit_kv.py --config configs/a6000_48gb_multiturn.yaml \\
        --rtypes rag code chat reason --rates 2 4 8 16 \\
        --duration 60 --out results_gpu_sweep/axis3

Runs len(arms) x len(rtypes) x len(rates) single-shot homogeneous-trace
probes drawn from the real corpora (--real-corpus is always on, matching
Axis-2's reasoning: acceptance and queueing behavior should come from actual
text, not synthetic templates). Output length is NOT forced, same as Axis-2.
Writes grid.json incrementally after every cell so a long run can resume.
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

STATIC_CAP_LO = 16
STATIC_CAP_HI = 256
GAMMA_STATIC = 4
GAMMA_FLOOR = 1

ARMS = {
    "static-cap-lo": {
        "controller": lambda cap_hi, gamma_init: {
            "spec": "static", "admit": "static", "coordination": "naive",
            "spec_kw": {"gamma": GAMMA_STATIC},
            "admit_kw": {"max_num_seqs": STATIC_CAP_LO},
        },
        "max_num_seqs_init": STATIC_CAP_LO,
    },
    "static-cap-hi": {
        "controller": lambda cap_hi, gamma_init: {
            "spec": "static", "admit": "static", "coordination": "naive",
            "spec_kw": {"gamma": GAMMA_STATIC},
            "admit_kw": {"max_num_seqs": cap_hi},
        },
        "max_num_seqs_init": STATIC_CAP_HI,
    },
    "adaptive-cap-only": {
        "controller": lambda cap_hi, gamma_init: {
            "spec": "static", "admit": "ttft-slack", "coordination": "naive",
            "spec_kw": {"gamma": GAMMA_FLOOR},
            "admit_kw": {"target_util": 0.7, "gain": 0.5, "period": 4,
                        "batch_min": 1, "batch_max": cap_hi, "init": gamma_init},
        },
        "max_num_seqs_init": STATIC_CAP_HI,
    },
    "kv-aware-cap-only": {
        "controller": lambda cap_hi, gamma_init: {
            "spec": "static", "admit": "kv-aware", "coordination": "naive",
            "spec_kw": {"gamma": GAMMA_FLOOR},
            "admit_kw": {"target_util": 0.7, "gain": 0.5, "period": 4,
                        "kv_target": 0.80, "kv_min_frac": 0.30,
                        "batch_min": 1, "batch_max": cap_hi, "init": gamma_init},
        },
        "max_num_seqs_init": STATIC_CAP_HI,
    },
    "full": {
        "controller": lambda cap_hi, gamma_init: {
            "spec": "gated", "admit": "kv-aware", "coordination": "naive",
            # threshold=0.15, not GatedSpec's own default of 0.15 left
            # implicit: measured mean_accept_rate in the first Axis-3 run was
            # [0.35, 0.46], and 0.45 (ClosedLoopSpec's original default)
            # solves the setpoint law to gamma~1 at those rates -- the gate
            # opened but never asked for anything above the floor. 0.15
            # targets gamma~2-2.4 at the same acceptance rates instead.
            "spec_kw": {"threshold": 0.15, "gain": 0.5, "deadband": 0.5,
                       "period": 4, "gamma_min": 0, "gamma_max": 8,
                       "gamma_init": GAMMA_STATIC, "gamma_floor": GAMMA_FLOOR,
                       "decode_bound_util": 0.85, "kv_headroom_frac": 0.15},
            "admit_kw": {"target_util": 0.7, "gain": 0.5, "period": 4,
                        "kv_target": 0.80, "kv_min_frac": 0.30,
                        "batch_min": 1, "batch_max": cap_hi, "init": gamma_init},
        },
        "max_num_seqs_init": STATIC_CAP_HI,
    },
}


def run_cell(base_cfg: dict, arm: str, rtype: str, rate: float,
            duration: float, seed: int, out_dir: str) -> dict:
    cfg = copy.deepcopy(base_cfg)
    spec = ARMS[arm]
    cfg["runtime"]["max_num_seqs_init"] = spec["max_num_seqs_init"]
    cfg["controller"] = spec["controller"](STATIC_CAP_HI, spec["max_num_seqs_init"])
    cfg_path = os.path.join(out_dir, f"cfg_{arm}_{rtype}_r{rate}.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    run_dir = os.path.join(out_dir, f"{arm}_{rtype}_r{rate}_s{seed}")
    cmd = [sys.executable, "-m", "specloop_rt.replay", "--config", cfg_path,
           "--trace", "homogeneous", "--rtype", rtype, "--rate", str(rate),
           "--duration", str(duration), "--seed", str(seed),
           "--real-corpus", "--out", run_dir]
    print(f">> arm={arm} rtype={rtype} rate={rate}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    m = summarize_run(run_dir, tpot_slo=base_cfg["runtime"]["tpot_slo_s"],
                      ttft_slo=base_cfg["runtime"]["ttft_slo_s"])
    return m


def main(argv=None):
    p = argparse.ArgumentParser("specloop-rt admission+KV controller sweep")
    p.add_argument("--config", required=True)
    p.add_argument("--rtypes", nargs="+", default=["rag", "code", "chat", "reason"])
    p.add_argument("--rates", type=float, nargs="+", default=[2, 4, 8, 16])
    p.add_argument("--arms", nargs="+", default=list(ARMS.keys()),
                   help="subset of arms to run, e.g. to resume/extend a partial sweep")
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results_gpu_sweep/axis3")
    a = p.parse_args(argv)

    for arm in a.arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; choices: {list(ARMS)}")

    os.makedirs(a.out, exist_ok=True)
    grid_path = os.path.join(a.out, "grid.json")
    grid = json.load(open(grid_path)) if os.path.exists(grid_path) else {}

    with open(a.config) as f:
        base_cfg = yaml.safe_load(f)

    for arm in a.arms:
        for rtype in a.rtypes:
            for rate in a.rates:
                key = f"{arm}_{rtype}_r{rate}"
                if key in grid and "error" not in grid[key]:
                    print(f".. skip {key} (already done)", flush=True)
                    continue
                try:
                    m = run_cell(base_cfg, arm, rtype, rate, a.duration, a.seed, a.out)
                    grid[key] = {"arm": arm, "rtype": rtype, "rate": rate, **m}
                except subprocess.CalledProcessError as e:
                    print(f"!! cell {key} FAILED: {e}", flush=True)
                    grid[key] = {"arm": arm, "rtype": rtype, "rate": rate, "error": str(e)}
                with open(grid_path, "w") as f:
                    json.dump(grid, f, indent=2)

    # ---- summary: mean SLO attainment / e2e_p95 / ITL-proxy per arm --------
    metrics = ["slo_attainment", "ttft_p99", "tpot_p99", "e2e_p95", "goodput_tok_s"]
    print("\n=== per-arm means across all (rtype, rate) cells ===")
    header = "arm".ljust(20) + "".join(m.rjust(16) for m in metrics)
    print(header)
    for arm in a.arms:
        rows = [grid[k] for k in grid if grid[k].get("arm") == arm and "error" not in grid[k]]
        if not rows:
            print(arm.ljust(20) + "no successful cells".rjust(16))
            continue
        cells = []
        for m in metrics:
            vals = [r[m] for r in rows if m in r and r[m] == r[m]]  # drop NaN
            cells.append(f"{(sum(vals) / len(vals)):>16.4f}" if vals else f"{'n/a':>16}")
        print(arm.ljust(20) + "".join(cells))

    n_failed = sum(1 for v in grid.values() if "error" in v)
    print(f"\n{len(grid) - n_failed}/{len(grid)} cells succeeded.")
    print(f"Full grid written to {grid_path}.")
    print("\nRead order for the write-up: static-cap-hi is the Axis-1/Axis-2 "
          "baseline. adaptive-cap-only minus static-cap-hi is the queueing-"
          "loop's contribution alone. kv-aware-cap-only minus adaptive-cap-"
          "only isolates the KV ceiling's incremental effect -- expect this "
          "gap to show up mainly in e2e_p95, not slo_attainment, since KV "
          "pressure's failure mode is preemption/recompute tail latency, not "
          "steady-state throughput. full minus kv-aware-cap-only isolates "
          "gated gamma's incremental effect -- per Axis-2, expect this gap to "
          "be small; if it is not small, that itself is a finding worth "
          "flagging rather than assuming a bug.")


if __name__ == "__main__":
    main()
