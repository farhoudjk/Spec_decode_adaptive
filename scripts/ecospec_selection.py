from __future__ import annotations


def greedy_ecospec_select(candidate_expert_sets: list, gamma: int) -> list:
    eps = 1e-6
    remaining = list(range(len(candidate_expert_sets)))
    buffer = set()
    selected = []
    while remaining and len(selected) < gamma:
        best_idx, best_score = None, -1.0
        for idx in remaining:
            new_experts = candidate_expert_sets[idx] - buffer
            delta_cost = len(new_experts)
            score = 1.0 / (delta_cost + eps)
            if score > best_score:
                best_score, best_idx = score, idx
        selected.append(best_idx)
        buffer |= candidate_expert_sets[best_idx]
        remaining.remove(best_idx)
    return selected


def confidence_only_select(gamma: int, pool_size: int) -> list:
    return list(range(min(gamma, pool_size)))


def expert_set_for_position(topk_per_position_step: list, pos: int) -> set:
    experts = set()
    for layer_experts in topk_per_position_step[pos]:
        experts.update(e for e in layer_experts if e >= 0)
    return experts
