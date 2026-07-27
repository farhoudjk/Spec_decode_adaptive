"""Axis-1 sweep: locate the admission-bound / decode-bound crossover.

Gates the entire regime-map investigation (see specloop_rt/README.md's
"Regime map" section once written): no controller comparison means anything
until we know a (rate, output_length) cell puts TPOT, not TTFT/queueing, in
the driver's seat. Uses a single fixed-gamma static controller throughout so
the sweep measures a property of the workload/hardware, not of any adaptive
policy -- see PROVENANCE.md-style reasoning in the module docstring below.

Usage:
    python3 scripts/sweep_boundedness.py --config configs/a6000_48gb_multiturn.yaml \\
        --rates 8 4 2 1 0.5 --output-lens 64 256 512 1024 \\
        --duration 60 --gamma 4 --out results_gpu_sweep/boundedness

Runs len(rates) x len(output_lens) single-shot homogeneous-trace probes
(default rtype: chat -- swap with --rtype once the axis-1 result is in hand
and axis-2 workload-type sweeps begin), each with ignore_eos forced on so
every request actually reaches the requested output length. Prints a grid of
bound_ratio and regime classification, and writes the full per-cell metrics
to <out>/grid.json for the crossover-contour plot.
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


def run_cell(base_cfg: dict, gamma: int, rate: float, out_tokens: int,
            duration: float, seed: int, out_dir: str, rtype: str) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["controller"] = {"spec": "static", "admit": "static",
                         "coordination": "naive",
                         "spec_kw": {"gamma": gamma},
                         "admit_kw": {"max_num_seqs": cfg["runtime"].get("max_num_seqs_init", 64)}}
    cfg["runtime"]["ignore_eos"] = True
    cfg_path = os.path.join(out_dir, f"cfg_r{rate}_o{out_tokens}.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    run_dir = os.path.join(out_dir, f"r{rate}_o{out_tokens}_s{seed}")
    cmd = [sys.executable, "-m", "specloop_rt.replay", "--config", cfg_path,
           "--trace", "homogeneous", "--rtype", rtype, "--rate", str(rate),
           "--duration", str(duration), "--seed", str(seed),
           "--force-output-tokens", str(out_tokens), "--out", run_dir]
    print(f">> rate={rate} out_tokens={out_tokens}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    m = summarize_run(run_dir, tpot_slo=base_cfg["runtime"]["tpot_slo_s"],
                      ttft_slo=base_cfg["runtime"]["ttft_slo_s"])
    return m


def main(argv=None):
    p = argparse.ArgumentParser("specloop-rt boundedness sweep")
    p.add_argument("--config", required=True)
    p.add_argument("--rates", type=float, nargs="+", default=[8, 4, 2, 1, 0.5])
    p.add_argument("--output-lens", type=int, nargs="+", default=[64, 256, 512, 1024])
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results_gpu_sweep/boundedness")
    a = p.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    with open(a.config) as f:
        base_cfg = yaml.safe_load(f)

    grid = {}
    for rate in a.rates:
        for out_tokens in a.output_lens:
            key = f"r{rate}_o{out_tokens}"
            try:
                m = run_cell(base_cfg, a.gamma, rate, out_tokens, a.duration, a.seed, a.out, rtype="chat")
                grid[key] = {"rate": rate, "output_tokens": out_tokens, **m}
            except subprocess.CalledProcessError as e:
                print(f"!! cell {key} FAILED: {e}", flush=True)
                grid[key] = {"rate": rate, "output_tokens": out_tokens, "error": str(e)}

    with open(os.path.join(a.out, "grid.json"), "w") as f:
        json.dump(grid, f, indent=2)

    print("\n=== bound_ratio grid (rows=rate, cols=output_tokens) ===")
    print("rate\\len  " + "  ".join(f"{ol:>8}" for ol in a.output_lens))
    for rate in a.rates:
        cells = []
        for ol in a.output_lens:
            g = grid.get(f"r{rate}_o{ol}", {})
            r = g.get("bound_ratio")
            cells.append(f"{r:>8.2f}" if r is not None else "     n/a")
        print(f"{rate:>8}  " + "  ".join(cells))

    print("\n=== regime grid ===")
    print("rate\\len  " + "  ".join(f"{ol:>10}" for ol in a.output_lens))
    for rate in a.rates:
        cells = []
        for ol in a.output_lens:
            g = grid.get(f"r{rate}_o{ol}", {})
            cells.append(f"{g.get('regime', 'n/a'):>10}")
        print(f"{rate:>8}  " + "  ".join(cells))

    n_decode = sum(1 for g in grid.values() if g.get("regime") == "decode_bound")
    print(f"\n{n_decode}/{len(grid)} cells reached decode_bound "
          f"(bound_ratio < 0.67).")
    if n_decode == 0:
        print("No cell reached decode_bound at the swept rates/lengths. Per the "
              "regime-map plan: this IS the headline finding if it holds after "
              "widening the sweep -- on this hardware/model/draft-method, "
              "speculation control may be irrelevant end-to-end. Before "
              "concluding that, widen --rates downward and --output-lens "
              "upward; the crossover may simply be outside this grid.")


if __name__ == "__main__":
    main()
