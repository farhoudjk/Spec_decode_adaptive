"""C2: fit the C_e*E(tree) cost term and check whether it explains axis6's
MoE within-cell residual. See AXIS7.md#13 for why this is the next step
after C1 confirmed the expert-activation mechanism is real (session 3,
sglang_verify_footprint_grid).

WHAT THIS JOINS: axis6's timing grid
(results_gpu_sweep/sglang_depth_width_rate_qwen3moe/grid.json, 288 cells,
full B x rate x D x W) against this session's verify-batch footprint grid
(results_gpu_sweep/sglang_verify_footprint_grid/grid.json, 9 cells,
D in {1,3,6} x W in {1,4,8}, fixed B=16 rate=4 -- see that sweep's module
docstring for why B x rate wasn't varied). The join key is
(num_steps, eagle_topk); on the footprint side there is exactly one row
per key (no B/rate axis), so this script pulls axis6's MATCHING B=16,rate=4
cell for a clean like-for-like comparison, not an average over axis6's
full load grid -- averaging would mix in load-driven cost variation that
has nothing to do with E(tree) and would bias C_e.

CAVEAT (see AXIS7.md#14, carried over here on purpose): this only checks 9
of axis6's 32 (D,W) combinations. Treat any fitted C_e as provisional until
the verify-footprint grid is extended to the full D in {1,2,3,4,6,8} x
W in {1,2,4,8} axis6 covers. Also: axis6's B=16,rate=4 cell and this
session's B=16,rate=4 cell come from DIFFERENT sweep runs (different
wall-clock sessions) -- this repo's own measurement notes (README.md
"Bugs fixed during calibration") document session-to-session throughput
drift of up to ~8%; a small, consistent-sign residual shift across all 9
cells could be that drift, not a load-model gap. This script does not
correct for it -- flags it in the printed output instead so it's visible,
not silently absorbed into the C_e fit.

MODEL FIT, mirroring axis6's own D+W affine / +interaction comparison
(reproduced at the top of this session's transcript, and matching
AXIS6.md's methodology): for the 9 matched cells,
  T_step = per_token_s_p50 * mean_accept_length      (verify-step cost, axis6's own convention)
  nodes-only:     T_step ~ D*W
  D+W affine:     T_step ~ D + W
  +interaction:   T_step ~ D + W + D*W
  +expert term:   T_step ~ D + W + E                 (E = mean_distinct_experts or mean_imbalance)
  +interaction+E: T_step ~ D + W + D*W + E
R^2 for each is printed side by side. The headline check: does +expert term
recover most of what +interaction bought (0.78 -> 0.86 in the original
288-cell fit), and does C_w shrink once E is added (the design brief's
"width cost was really expert cost in disguise" prediction)?
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def load_grid(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def r2_fit(X: np.ndarray, y: np.ndarray):
    """OLS R^2 and coefficients for y ~ X + intercept."""
    X1 = np.column_stack([X, np.ones(len(y))])
    coef, _, _, _ = np.linalg.lstsq(X1, y, rcond=None)
    pred = X1 @ coef
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return r2, coef


def build_matched_rows(axis6_grid: dict, footprint_grid: dict, B: int, rate: float,
                        rtype: str) -> list:
    rows = []
    for key, fp in footprint_grid.items():
        D = fp["num_steps"]
        W = fp["eagle_topk"]
        axis6_key = f"steps{D}_topk{W}_B{B}_rate{rate:g}_{rtype}"
        a6 = axis6_grid.get(axis6_key)
        if a6 is None or "error" in a6:
            print(f"WARNING: no matching axis6 cell for {axis6_key} -- skipping")
            continue
        if "error" in fp:
            print(f"WARNING: footprint cell {key} has an error -- skipping")
            continue
        rows.append({
            "D": D, "W": W,
            "T_step": a6["per_token_s_p50"] * a6["mean_accept_length"],
            "axis6_per_token_s_p50": a6["per_token_s_p50"],
            "axis6_mean_accept_length": a6["mean_accept_length"],
            "mean_distinct_experts": fp["mean_distinct_experts"],
            "mean_imbalance": fp["mean_imbalance"],
            "mean_verify_batch_tokens": fp["mean_verify_batch_tokens"],
        })
    return rows


def main(argv=None):
    p = argparse.ArgumentParser("Fit C_e*E(tree) cost term against axis6's residual")
    p.add_argument("--axis6-grid",
                    default="results_gpu_sweep/sglang_depth_width_rate_qwen3moe/grid.json")
    p.add_argument("--footprint-grid",
                    default="results_gpu_sweep/sglang_verify_footprint_grid/grid.json")
    p.add_argument("--B", type=int, default=16,
                    help="axis6 batch cap to match against (must equal the footprint "
                         "grid's fixed B -- see that sweep's --Bs default)")
    p.add_argument("--rate", type=float, default=4.0,
                    help="axis6 arrival rate to match against (must equal the footprint "
                         "grid's fixed rate)")
    p.add_argument("--rtype", default="code")
    p.add_argument("--e-field", default="mean_distinct_experts",
                    choices=["mean_distinct_experts", "mean_imbalance"],
                    help="which footprint statistic to use as E(tree)")
    a = p.parse_args(argv)

    if not os.path.exists(a.axis6_grid):
        raise SystemExit(f"axis6 grid not found: {a.axis6_grid}")
    if not os.path.exists(a.footprint_grid):
        raise SystemExit(f"footprint grid not found: {a.footprint_grid}")

    axis6_grid = load_grid(a.axis6_grid)
    footprint_grid = load_grid(a.footprint_grid)

    rows = build_matched_rows(axis6_grid, footprint_grid, a.B, a.rate, a.rtype)
    if len(rows) < 5:
        raise SystemExit(f"only {len(rows)} matched cells -- too few to fit "
                         f"(need at least ~5 for a 3-4 parameter model); check "
                         f"--B/--rate match the footprint grid's fixed load point")

    print(f"Matched {len(rows)} cells (D,W) between axis6 (B={a.B}, rate={a.rate}) "
          f"and the verify-footprint grid.\n")

    D = np.array([r["D"] for r in rows], dtype=float)
    W = np.array([r["W"] for r in rows], dtype=float)
    T = np.array([r["T_step"] for r in rows], dtype=float)
    E = np.array([r[a.e_field] for r in rows], dtype=float)
    nodes = D * W

    print(f"{'D':>3} {'W':>3} {'T_step':>10} {a.e_field:>22}")
    for r in rows:
        print(f"{r['D']:>3} {r['W']:>3} {r['T_step']:>10.5f} {r[a.e_field]:>22.3f}")
    print()

    r2_nodes, _ = r2_fit(nodes.reshape(-1, 1), T)
    r2_dw, coef_dw = r2_fit(np.column_stack([D, W]), T)
    r2_int, coef_int = r2_fit(np.column_stack([D, W, nodes]), T)
    r2_e, coef_e = r2_fit(np.column_stack([D, W, E]), T)
    r2_int_e, coef_int_e = r2_fit(np.column_stack([D, W, nodes, E]), T)

    print("=== Within-cell cost-model R^2 ({:d}-cell match, B={:d} rate={:g}) ===".format(len(rows), a.B, a.rate))
    print(f"nodes-only (D*W):              R2={r2_nodes:.4f}")
    print(f"D+W affine:                    R2={r2_dw:.4f}   C_d={coef_dw[0]:.6f} C_w={coef_dw[1]:.6f}")
    print(f"D+W+interaction:                R2={r2_int:.4f}   C_d={coef_int[0]:.6f} C_w={coef_int[1]:.6f} C_dw={coef_int[2]:.6f}")
    print(f"D+W+E ({a.e_field}):  R2={r2_e:.4f}   C_d={coef_e[0]:.6f} C_w={coef_e[1]:.6f} C_e={coef_e[2]:.6f}")
    print(f"D+W+interaction+E:              R2={r2_int_e:.4f}   C_d={coef_int_e[0]:.6f} C_w={coef_int_e[1]:.6f} C_dw={coef_int_e[2]:.6f} C_e={coef_int_e[3]:.6f}")
    print()

    print("=== Headline checks ===")
    print(f"Does +E recover what +interaction bought? "
          f"interaction gain={r2_int - r2_dw:+.4f}, E gain={r2_e - r2_dw:+.4f} "
          f"({'E explains most of it' if r2_e - r2_dw >= 0.7 * (r2_int - r2_dw) else 'E falls short of the interaction-term gain'})")
    cw_shrink = coef_dw[1] - coef_e[1]
    print(f"Does C_w shrink once E is added? "
          f"C_w (D+W only)={coef_dw[1]:.6f} -> C_w (D+W+E)={coef_e[1]:.6f} "
          f"({'shrinks' if cw_shrink > 0 else 'does not shrink'}, delta={cw_shrink:+.6f})")
    print()
    print("REMINDER (see module docstring): axis6's B16/rate4 cell and this "
          "session's footprint cell are from DIFFERENT sweep runs -- this "
          "repo's own notes document up to ~8% session-to-session throughput "
          "drift. A small uniform-sign residual shift is not automatically "
          "evidence for or against C_e; look at whether the SHAPE of the "
          "residual (not just its magnitude) tracks E across cells.")
    print()
    axis6_dw_pairs = {(v["num_steps"], v["eagle_topk"]) for v in axis6_grid.values()
                       if v.get("B") == a.B and v.get("rate") == a.rate and "error" not in v}
    matched_dw_pairs = {(r["D"], r["W"]) for r in rows}
    missing = sorted(axis6_dw_pairs - matched_dw_pairs)
    if missing:
        print(f"CAVEAT: {len(matched_dw_pairs)}/{len(axis6_dw_pairs)} of axis6's (D,W) "
              f"cells at B={a.B},rate={a.rate} are covered -- missing: {missing}. "
              f"Treat any fitted C_e as provisional until the verify-footprint grid "
              f"covers the remaining cells.")
    else:
        print(f"Full coverage: all {len(matched_dw_pairs)} of axis6's (D,W) cells at "
              f"B={a.B},rate={a.rate} are matched. Still only ONE (B,rate) load point "
              f"and ONE workload ({a.rtype}) -- see AXIS7.md#14 for what's still open "
              f"(other load points, other workloads).")


if __name__ == "__main__":
    main()
