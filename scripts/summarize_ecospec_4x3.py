from __future__ import annotations
import argparse
import glob
import json
import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPTS_DIR)
from analyze_ecospec_concurrent import compare_one_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results_gpu_sweep/ecospec_4x3")
    ap.add_argument("--gamma", type=int, default=3)
    ap.add_argument("--tag", default="D2W4")
    ap.add_argument("--out", default="results_gpu_sweep/ecospec_4x3/summary.json")
    a = ap.parse_args()

    results = {}
    cells = sorted(glob.glob(os.path.join(a.root, "B*_r*")))
    print(f"Found {len(cells)} cell directories\n")
    for cell_dir in cells:
        base = os.path.basename(cell_dir)
        verify_file = os.path.join(cell_dir, f"verify_{a.tag}.jsonl")
        if not os.path.exists(verify_file):
            print(f"  {base}: MISSING {verify_file}")
            continue
        stats = compare_one_file(verify_file, a.gamma)
        results[base] = stats
        if stats.get("n_steps"):
            print(f"  {base}: n_steps={stats['n_steps']} "
                  f"conf={stats['conf_only_mean_experts']:.2f} "
                  f"eco={stats['ecospec_mean_experts']:.2f} "
                  f"reduction={stats['ecospec_reduction_pct']:+.2f}%")
        else:
            print(f"  {base}: no usable verify steps "
                  f"({stats.get('n_untagged_steps_skipped', 0)} untagged)")

    json.dump(results, open(a.out, "w"), indent=2)
    print(f"\nWrote {a.out}")

    valid = [v for v in results.values() if v.get("n_steps")]
    if valid:
        mean_reduction = sum(v["ecospec_reduction_pct"] for v in valid) / len(valid)
        print(f"\nMean EcoSpec expert-footprint reduction across {len(valid)} load points: "
              f"{mean_reduction:+.2f}%")


if __name__ == "__main__":
    main()
