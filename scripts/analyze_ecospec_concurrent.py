from __future__ import annotations

import argparse
import json
from collections import defaultdict

from ecospec_selection import (
    greedy_ecospec_select,
    confidence_only_select,
    expert_set_for_position,
)


def compare_one_file(verify_path: str, gamma: int) -> dict:
    total_steps = 0
    conf_total = eco_total = 0
    beats = ties = worse = 0
    skipped_untagged = 0

    with open(verify_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            step = json.loads(line)
            tags = step.get("req_tag_per_position")
            topk_per_pos = step["topk_per_position"]
            if tags is None:
                skipped_untagged += 1
                continue

            by_req = defaultdict(list)
            for pos, tag in enumerate(tags):
                by_req[tag].append(pos)

            for tag, positions in by_req.items():
                pool_size = len(positions)
                if pool_size <= gamma:
                    continue
                total_steps += 1
                expert_sets = [expert_set_for_position(topk_per_pos, p) for p in positions]

                conf_idx = confidence_only_select(gamma, pool_size)
                conf_experts = set()
                for i in conf_idx:
                    conf_experts |= expert_sets[i]

                eco_idx = greedy_ecospec_select(expert_sets, gamma)
                eco_experts = set()
                for i in eco_idx:
                    eco_experts |= expert_sets[i]

                conf_total += len(conf_experts)
                eco_total += len(eco_experts)
                if len(eco_experts) < len(conf_experts):
                    beats += 1
                elif len(eco_experts) == len(conf_experts):
                    ties += 1
                else:
                    worse += 1

    if total_steps == 0:
        return {"n_steps": 0, "n_untagged_steps_skipped": skipped_untagged}
    return {
        "n_steps": total_steps,
        "n_untagged_steps_skipped": skipped_untagged,
        "conf_only_mean_experts": conf_total / total_steps,
        "ecospec_mean_experts": eco_total / total_steps,
        "ecospec_reduction_pct": 100 * (1 - eco_total / conf_total) if conf_total else 0.0,
        "ecospec_beats_pct": 100 * beats / total_steps,
        "ecospec_ties_pct": 100 * ties / total_steps,
        "ecospec_worse_pct": 100 * worse / total_steps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-file", required=True)
    ap.add_argument("--gamma", type=int, default=3)
    a = ap.parse_args()

    stats = compare_one_file(a.verify_file, a.gamma)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
