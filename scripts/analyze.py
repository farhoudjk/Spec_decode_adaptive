"""Aggregate summary.txt over every run directory produced by run_matrix.sh.

Usage: python3 scripts/analyze.py <results_dir> --tpot-slo 0.040 --ttft-slo 2.0

Each subdirectory of <results_dir> matching "<arm>_s<seed>" (as written by
scripts/run_matrix.sh's mkvar/replay loop) is summarized with
specloop_rt.analysis.summarize_run, then rows sharing an arm name are averaged
across seeds. The coordination arms (coord_naive/coord_timescale/
coord_hysteresis) ran on the step-perturbation trace, so settling metrics are
computed with t_event at the trace's warmup/post split recorded in that run's
steps.jsonl meta; core arms have no perturbation event.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np

# run as `python3 scripts/analyze.py`, so the repo root (this file's parent's
# parent) needs to be on sys.path for the specloop_rt import to resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from specloop_rt.analysis import load_run, summarize_run

RUN_RE = re.compile(r"^(?P<arm>.+)_s(?P<seed>\d+)$")

COLUMNS = [
    ("slo_attainment", "SLO attain.", "{:.1%}"),
    ("goodput_tok_s", "Goodput (tok/s)", "{:.0f}"),
    ("tpot_p99", "TPOT p99", "{:.4f}s"),
    ("ttft_p99", "TTFT p99", "{:.1f}s"),
    ("mean_gamma", "Mean gamma", "{:.2f}"),
    ("mean_accept_rate", "Mean accept", "{:.1%}"),
    ("gamma_spectral_concentration", "Spectral conc.", "{:.3f}"),
    ("gamma_cv", "Gamma CV", "{:.3f}"),
]


def main(argv=None):
    p = argparse.ArgumentParser("specloop-rt analyze")
    p.add_argument("results_dir")
    p.add_argument("--tpot-slo", type=float, default=0.040)
    p.add_argument("--ttft-slo", type=float, default=2.0)
    a = p.parse_args(argv)

    by_arm = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(a.results_dir, "*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        m = RUN_RE.match(name)
        if not m:
            continue
        if not os.path.exists(os.path.join(d, "steps.jsonl")):
            continue
        arm = m.group("arm")
        is_coord = arm.startswith("coord_")
        t_event = None
        if is_coord:
            run = load_run(d)
            s = run["steps"]
            if not s.empty:
                t0 = s.t_wall.min()
                t1 = s.t_wall.max()
                t_event = t0 + (t1 - t0) / 3.0
        try:
            m_ = summarize_run(d, a.tpot_slo, a.ttft_slo, t_event=t_event)
        except FileNotFoundError:
            print(f"skip {d}: missing steps.jsonl/requests.jsonl")
            continue
        by_arm[arm].append(m_)

    if not by_arm:
        print(f"no runs found under {a.results_dir}")
        return

    rows = []
    for arm, runs in by_arm.items():
        row = {"arm": arm, "n_seeds": len(runs)}
        for key, _, _ in COLUMNS:
            vals = [r[key] for r in runs if key in r and r[key] == r[key]]  # drop NaN
            row[key] = float(np.mean(vals)) if vals else float("nan")
        row["oscillatory"] = bool(np.mean([r.get("oscillatory", False) for r in runs]) >= 0.5)
        rows.append(row)

    name_w = max(len("arm"), max(len(r["arm"]) for r in rows))
    header = f"{'arm':<{name_w}}  " + "  ".join(h for _, h, _ in COLUMNS) + "  Stability"
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda r: r["arm"]):
        cells = [fmt.format(r[key]) if r[key] == r[key] else "n/a" for key, _, fmt in COLUMNS]
        stability = "oscillatory" if r["oscillatory"] else "stable"
        print(f"{r['arm']:<{name_w}}  " + "  ".join(cells) + f"  {stability}  (n={r['n_seeds']})")


if __name__ == "__main__":
    main()
