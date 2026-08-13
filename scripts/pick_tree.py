"""C3a: expert-footprint-corrected tree-shape picker for MoE EAGLE3 serving.

Given (B, rate), picks (D, W) using the design brief's marginal-utility
rule (§4): grow the axis (D or W) with the larger marginal
acceptance-per-marginal-cost, stop when both fall below the current
average. This is a CONFIG-TIME picker, not a live controller -- SGLang's
tree width is launch-time-static (AXIS6.md §0, re-confirmed this session:
`adaptive_spec_params.py`'s `adaptive_unsupported_reason()` hard-refuses
`--speculative-adaptive` unless `eagle_topk in (None, 1)`). pick_tree()
is meant to run BEFORE `sglang.launch_server`, not during serving.

MODEL: per-load-point lookup, NOT one global cross-load formula. See
AXIS7.md §16-§17 for the full fitting trace and why this design was
chosen over the alternatives tried first:

  ATTEMPT 1 (single global fit, all 3 load points combined): D+W+E OLS
  gave R²=0.889 but with NEGATIVE C_d, C_w (collinearity -- D,W,E aren't
  independent once load varies, since E is itself downstream of D,W,B,rate
  via verify_batch_tokens). A picker built on this predicted NEGATIVE
  T_step at 2 of the 3 measured load points, breaking the marginal-utility
  math (division by a near-zero/negative denominator, picker stalled at
  D=1,W=1 instead of making a real choice).

  ATTEMPT 2 (global fit, C_d/C_w constrained >=0 via nnls): constraint
  drove BOTH to exactly zero (R²=0.830, E-only). Still produced negative
  T_step at light/mid load, because a single intercept can't simultaneously
  fit the ~4x range in absolute T_step scale across B=8..32,rate=2..12 --
  the fitted intercept that works for heavy load is wrong for light load
  and vice versa.

  ATTEMPT 3 (this module): fit T_step ~ D+W+E SEPARATELY at each of the 3
  GPU-measured (B,rate) points (the same per-load-point convention AXIS7.md
  §14/§15 already use, NOT the cross-load attempt above). Light and mid
  load fits have all-positive coefficients with no correction needed
  (R²=0.974, 0.947). Heavy load's plain OLS fit had the same
  negative-coefficient problem as the global fit (collinearity is worse at
  high load, more concurrent-request coalescing variance) -- fixed the same
  way, nnls-constraining C_d,C_w>=0, which dropped C_d to 0 and kept
  R²=0.817 (down from 0.892 unconstrained, but with a sane, always-positive
  T_step prediction, verified over the full D×W grid).

  pick_tree(B, rate) requires (B, rate) to exactly match one of the 3
  measured points -- NO interpolation or extrapolation to untested load is
  attempted. That is a real limitation (see AXIS7.md §16.4): a genuine
  C_e(B,rate) load-scaling law, fit the way axis6 fits C_d,C_w ∝
  λ^1.4·B^-0.15, would need more than 3 (B,rate) points to fit honestly
  and is future work, not done here.

Acceptance (axis6, MoE, structure-only -- AXIS6.md, reproduced exactly
this session): ℓ(D,W) = L·(1 − r^D)·(1 + g·log2(W))
  L=3.448, r=0.576, g=0.063 (R²=0.953 on the original 24-cell fit)
"""
from __future__ import annotations

import argparse
import math

# --- Acceptance model (axis6, MoE, structure-only) ---
ACC_L = 3.448
ACC_R = 0.576
ACC_G = 0.063

# --- Expert-footprint model, per (D,W), fit separately at each measured
# load point (AXIS7.md §12/§16 raw mean_distinct_experts tables) --
# reproduced here as a lookup rather than a formula, since E(D,W) does
# NOT reduce to a clean closed form even within one load point (it's a
# measured GPU quantity, not derived) ---
MEASURED_E = {
    (8, 2.0): {
        (1, 1): 18.48, (1, 4): 28.67, (1, 8): 35.73,
        (3, 1): 29.33, (3, 4): 41.40, (3, 8): 44.30,
        (6, 1): 36.49, (6, 4): 45.49, (6, 8): 44.88,
    },
    (16, 4.0): {
        (1, 1): 22.35, (1, 2): 30.77, (1, 4): 35.58, (1, 8): 45.06,
        (2, 1): 28.01, (2, 2): 33.92, (2, 4): 42.26, (2, 8): 51.36,
        (3, 1): 31.29, (3, 2): 38.86, (3, 4): 47.13, (3, 8): 49.84,
        (4, 1): 36.75, (4, 2): 42.92, (4, 4): 52.45, (4, 8): 50.92,
        (6, 1): 42.86, (6, 2): 49.88, (6, 4): 50.73, (6, 8): 50.37,
        (8, 1): 48.33, (8, 2): 60.52, (8, 4): 50.67, (8, 8): 50.61,
    },
    (32, 12.0): {
        (1, 1): 57.74, (1, 4): 74.60, (1, 8): 85.34,
        (3, 1): 65.96, (3, 4): 83.56, (3, 8): 86.47,
        (6, 1): 77.42, (6, 4): 85.38, (6, 8): 85.59,
    },
}

# --- Cost model T_step(D,W,E) = C_d*D + C_w*W + C_e*E + C0, fit
# separately per load point. Light/mid: plain OLS, all coefficients
# already non-negative. Heavy: nnls-constrained (see module docstring
# ATTEMPT 3) -- C_d dropped to exactly 0. ---
COST_MODEL = {
    (8, 2.0):   {"C_d": 0.000669, "C_w": 0.000400, "C_e": 0.000357, "C0": 0.008774},
    (16, 4.0):  {"C_d": 0.000395, "C_w": 0.000518, "C_e": 0.000456, "C0": 0.012467},
    (32, 12.0): {"C_d": 0.000000, "C_w": 0.003295, "C_e": 0.002941, "C0": -0.113136},
}

SUPPORTED_LOAD_POINTS = sorted(COST_MODEL.keys())


def acceptance(D: float, W: float) -> float:
    return ACC_L * (1 - ACC_R ** D) * (1 + ACC_G * math.log2(W))


def _lookup_E(load_point: tuple, D: int, W: int) -> float:
    table = MEASURED_E[load_point]
    if (D, W) in table:
        return table[(D, W)]
    # Not every (D,W) was measured at every load point (light/heavy grids
    # only cover W in {1,4,8}, not the full W in {1,2,4,8} mid-load has).
    # Interpolate log-linearly in W (matches the saturating-with-W shape
    # seen throughout AXIS7.md's tables) rather than refuse -- but this is
    # an approximation the caller should know about.
    if W == 2:
        w1 = table.get((D, 1))
        w4 = table.get((D, 4))
        if w1 is not None and w4 is not None:
            # log-linear interpolation between W=1 and W=4 (2 doublings apart)
            return w1 * (w4 / w1) ** 0.5
    raise KeyError(f"No measured or interpolable E for D={D}, W={W} at {load_point}")


def predicted_T_step(load_point: tuple, D: int, W: int) -> float:
    E = _lookup_E(load_point, D, W)
    m = COST_MODEL[load_point]
    return m["C_d"] * D + m["C_w"] * W + m["C_e"] * E + m["C0"]


def pick_tree(B: int, rate: float, verbose: bool = False) -> tuple[int, int]:
    """Marginal-utility tree-shape picker (design brief §4). (B, rate)
    must exactly match one of SUPPORTED_LOAD_POINTS -- see module
    docstring for why this doesn't interpolate/extrapolate across load."""
    load_point = (B, rate)
    if load_point not in COST_MODEL:
        raise ValueError(
            f"(B={B}, rate={rate}) not in SUPPORTED_LOAD_POINTS={SUPPORTED_LOAD_POINTS}. "
            f"This picker only supports the 3 GPU-measured load points -- see module "
            f"docstring for why cross-load extrapolation isn't attempted."
        )

    valid_dw = sorted(MEASURED_E[load_point].keys())
    valid_D = sorted(set(d for d, w in valid_dw))
    valid_W = sorted(set(w for d, w in valid_dw))

    D_idx, W_idx = 0, 0
    D, W = valid_D[D_idx], valid_W[W_idx]

    while True:
        try:
            ell = acceptance(D, W)
            T = predicted_T_step(load_point, D, W)
        except KeyError:
            break
        avg = ell / T if T > 0 else float("inf")

        can_grow_D = D_idx + 1 < len(valid_D)
        can_grow_W = W_idx + 1 < len(valid_W)
        if not can_grow_D and not can_grow_W:
            break

        m_D = -float("inf")
        if can_grow_D:
            try:
                D_next = valid_D[D_idx + 1]
                ell_next = acceptance(D_next, W)
                T_next = predicted_T_step(load_point, D_next, W)
                m_D = (ell_next - ell) / (T_next - T) if T_next != T else float("inf")
            except KeyError:
                can_grow_D = False

        m_W = -float("inf")
        if can_grow_W:
            try:
                W_next = valid_W[W_idx + 1]
                ell_next = acceptance(D, W_next)
                T_next = predicted_T_step(load_point, D, W_next)
                m_W = (ell_next - ell) / (T_next - T) if T_next != T else float("inf")
            except KeyError:
                can_grow_W = False

        if not can_grow_D and not can_grow_W:
            break

        best_m = max(m_D, m_W)
        if best_m <= avg:
            break

        if verbose:
            print(f"D={D} W={W} avg={avg:.4f} m_D={m_D:.4f} m_W={m_W:.4f} "
                  f"-> grow {'D' if m_D >= m_W else 'W'}")

        if m_D >= m_W:
            D_idx += 1
        else:
            W_idx += 1
        D, W = valid_D[D_idx], valid_W[W_idx]

    return D, W


def main(argv=None):
    p = argparse.ArgumentParser("C3a: expert-footprint-corrected tree-shape picker")
    p.add_argument("--B", type=int, required=True)
    p.add_argument("--rate", type=float, required=True)
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)

    D, W = pick_tree(a.B, a.rate, verbose=a.verbose)
    T = predicted_T_step((a.B, a.rate), D, W)
    print(f"B={a.B} rate={a.rate}: pick D={D} W={W}")
    print(f"  predicted ell={acceptance(D,W):.3f} T_step={T:.5f}")


if __name__ == "__main__":
    main()
