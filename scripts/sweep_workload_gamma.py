"""Axis-2 sweep: does the optimal fixed gamma depend on workload type and rate?

Companion to scripts/sweep_boundedness.py (axis 1: rate x output_length ->
admission-bound vs decode-bound). Axis 1 fixed gamma and swept (rate, length)
to map the regime a controller has to survive. Axis 2 fixes the regime inputs
that matter least for THIS question (uses each rtype's natural output-length
distribution, not a forced length) and instead sweeps (rtype, gamma) at a few
rates, with a single static-gamma controller throughout -- no adaptation, so
any gamma-dependence measured here is a property of the workload/hardware,
not of ClosedLoopSpec or any other controller.

Hypothesis under test: a single fixed gamma is near-optimal for every
workload type and rate. If the goodput-maximizing (or SLO-attainment-
maximizing) gamma shifts across rtype in {rag, code, chat, reason} and/or
across rate, that is the empirical case for a workload/rate-aware adaptive
controller. If it does not shift, adaptive-by-workload-type is not worth
building.

Usage:
    python3 scripts/sweep_workload_gamma.py --config configs/a6000_48gb_multiturn.yaml \\
        --rtypes rag code chat reason --gammas 0 1 2 4 6 8 --rates 2 4 8 16 \\
        --duration 60 --out results_gpu_sweep/axis2

Runs len(rtypes) x len(gammas) x len(rates) single-shot homogeneous-trace
probes drawn from the real corpora (--real-corpus is always on: the point is
that acceptance behaviour comes from actual SQuAD/HumanEval/ShareGPT/
CNN-DailyMail text, not synthetic templates), each with a static gamma and a
generous static admission cap (same anti-confound as axis 1: this sweep asks
about speculation, not queueing). Output length is NOT forced -- each
corpus's natural length distribution is part of the effect being measured.
Writes the per-cell metrics to <out>/grid.json INCREMENTALLY -- after every
cell, not just at the end -- so a crash or manual stop partway through a long
sweep (each cell pays a full vLLM engine boot/compile, so the whole grid can
run for hours) does not lose already-completed cells. Rerunning the same
command resumes: any cell already present in grid.json with no "error" key is
skipped rather than redone. Prints, per rate, a (rtype x gamma) goodput table
with the argmax gamma marked per row.
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


def run_cell(base_cfg: dict, gamma: int, rate: float, rtype: str,
            duration: float, seed: int, out_dir: str, max_num_seqs: int) -> dict:
    cfg = copy.deepcopy(base_cfg)
    # generous static admission cap -- same reasoning as axis 1: this sweep
    # asks whether ACCEPTANCE/goodput varies with workload type and rate, not
    # whether the admission queue backs up.
    cfg["runtime"]["max_num_seqs_init"] = max_num_seqs
    cfg["controller"] = {"spec": "static", "admit": "static",
                         "coordination": "naive",
                         "spec_kw": {"gamma": gamma},
                         "admit_kw": {"max_num_seqs": max_num_seqs}}
    cfg_path = os.path.join(out_dir, f"cfg_{rtype}_g{gamma}_r{rate}.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    run_dir = os.path.join(out_dir, f"{rtype}_g{gamma}_r{rate}_s{seed}")
    cmd = [sys.executable, "-m", "specloop_rt.replay", "--config", cfg_path,
           "--trace", "homogeneous", "--rtype", rtype, "--rate", str(rate),
           "--duration", str(duration), "--seed", str(seed),
           "--real-corpus", "--out", run_dir]
    print(f">> rtype={rtype} gamma={gamma} rate={rate}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    m = summarize_run(run_dir, tpot_slo=base_cfg["runtime"]["tpot_slo_s"],
                      ttft_slo=base_cfg["runtime"]["ttft_slo_s"])
    return m


def main(argv=None):
    p = argparse.ArgumentParser("specloop-rt workload x gamma sweep (axis 2)")
    p.add_argument("--config", required=True)
    p.add_argument("--rtypes", nargs="+", default=["rag", "code", "chat", "reason"],
                   help="rag=SQuAD, code=HumanEval, chat=ShareGPT, reason=CNN-DailyMail "
                        "(summarization -- the long-output/high-repetition case)")
    p.add_argument("--gammas", type=int, nargs="+", default=[0, 1, 2, 4, 6, 8])
    p.add_argument("--rates", type=float, nargs="+", default=[2, 4, 8, 16])
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-num-seqs", type=int, default=256,
                   help="admission cap, held generous by default (see axis-1 rationale) "
                        "so the sweep isolates the speculation/workload effect")
    p.add_argument("--out", default="results_gpu_sweep/axis2")
    a = p.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    with open(a.config) as f:
        base_cfg = yaml.safe_load(f)

    grid_path = os.path.join(a.out, "grid.json")
    grid = {}
    if os.path.exists(grid_path):
        with open(grid_path) as f:
            grid = json.load(f)
        n_done = sum(1 for v in grid.values() if "error" not in v)
        print(f">> resuming: {grid_path} has {n_done} completed cell(s) already; "
              f"they will be skipped.", flush=True)

    def save_grid():
        # write-then-rename so a kill mid-write never corrupts the last good grid.json
        tmp_path = grid_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(grid, f, indent=2)
        os.replace(tmp_path, grid_path)

    total = len(a.rates) * len(a.rtypes) * len(a.gammas)
    done = 0
    for rate in a.rates:
        for rtype in a.rtypes:
            for gamma in a.gammas:
                key = f"{rtype}_g{gamma}_r{rate}"
                done += 1
                if key in grid and "error" not in grid[key]:
                    print(f">> [{done}/{total}] skipping {key} (already completed)", flush=True)
                    continue
                print(f">> [{done}/{total}] {key}", flush=True)
                try:
                    m = run_cell(base_cfg, gamma, rate, rtype, a.duration, a.seed,
                                a.out, max_num_seqs=a.max_num_seqs)
                    grid[key] = {"rtype": rtype, "gamma": gamma, "rate": rate, **m}
                except subprocess.CalledProcessError as e:
                    print(f"!! cell {key} FAILED: {e}", flush=True)
                    grid[key] = {"rtype": rtype, "gamma": gamma, "rate": rate, "error": str(e)}
                save_grid()

    for metric in ("goodput_tok_s", "mean_accept_rate", "slo_attainment"):
        print(f"\n=== {metric} (rows=rtype, cols=gamma) ===")
        for rate in a.rates:
            print(f"-- rate={rate} --")
            print("rtype     " + "  ".join(f"{g:>8}" for g in a.gammas) + "   argmax_gamma")
            for rtype in a.rtypes:
                vals = []
                for gamma in a.gammas:
                    g = grid.get(f"{rtype}_g{gamma}_r{rate}", {})
                    v = g.get(metric)
                    vals.append(v)
                cells = [f"{v:>8.3f}" if v is not None else "     n/a" for v in vals]
                finite = [(g, v) for g, v in zip(a.gammas, vals) if v is not None]
                argmax_g = max(finite, key=lambda gv: gv[1])[0] if finite else None
                print(f"{rtype:<10}" + "  ".join(cells) + f"   {argmax_g}")

    print("\n=== argmax-gamma summary (rows=rtype, cols=rate) ===")
    print("rtype     " + "  ".join(f"{r:>6}" for r in a.rates))
    for rtype in a.rtypes:
        cells = []
        for rate in a.rates:
            finite = [(g, grid.get(f"{rtype}_g{g}_r{rate}", {}).get("goodput_tok_s"))
                     for g in a.gammas]
            finite = [(g, v) for g, v in finite if v is not None]
            argmax_g = max(finite, key=lambda gv: gv[1])[0] if finite else None
            cells.append(f"{argmax_g!s:>6}")
        print(f"{rtype:<10}" + "  ".join(cells))

    print("\nIf argmax_gamma is constant across this whole table, a single fixed "
          "gamma is sufficient and workload/rate-adaptive gamma is not "
          "empirically justified by this sweep. If it varies by rtype and/or "
          "by rate, that variation is the case for building the adaptive "
          "controller next -- proceed using mean_accept_rate's variation "
          "across cells as the signal an adaptive controller would need to "
          "sense.")


if __name__ == "__main__":
    main()
