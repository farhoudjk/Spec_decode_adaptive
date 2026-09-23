from __future__ import annotations
import json, math, os
import numpy as np
from scipy.optimize import nnls, curve_fit

GRID_PATH = "results_gpu_sweep/gptoss_4x3/grid.json"
OUT_PATH = "results_gpu_sweep/gptoss_4x3/b0_b1_b2_full.json"

def goodput(cell):
    return 1.0 / cell["per_token_s_p50"] if cell.get("per_token_s_p50") else 0.0

def fit_acceptance(cells):
    Ds = np.array([c["num_steps"] for c in cells], dtype=float)
    Ws = np.array([c["eagle_topk"] for c in cells], dtype=float)
    accs = np.array([c["mean_accept_length"] for c in cells], dtype=float)
    def model(X, L, r, g):
        D, W = X
        return L * (1 - r**D) * (1 + g * np.log2(W))
    try:
        popt, _ = curve_fit(model, (Ds, Ws), accs, p0=[2.0, 0.5, 0.1],
                             bounds=([0.1, 0.01, -1], [10, 0.99, 1]), maxfev=10000)
        return {"L": popt[0], "r": popt[1], "g": popt[2]}
    except Exception as e:
        print(f"WARNING: acceptance fit failed ({e}), using measured mean per (D,W) instead")
        return None

def acceptance_lookup(cells):
    by_dw = {}
    for c in cells:
        k = (c["num_steps"], c["eagle_topk"])
        by_dw.setdefault(k, []).append(c["mean_accept_length"])
    return {k: sum(v)/len(v) for k, v in by_dw.items()}

def fit_cost_model(shapes_at_lp, use_E):
    rows, ys = [], []
    for (D, W), c in shapes_at_lp.items():
        T_step = c["per_token_s_p50"] * c["mean_accept_length"]
        feat = [D, W, c["mean_distinct_experts"], 1.0] if use_E else [D, W, 1.0]
        rows.append(feat)
        ys.append(T_step)
    X = np.array(rows)
    y = np.array(ys)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    if (coef[:-1] < 0).any():
        coef, _ = nnls(X, y)
    keys = (["C_d", "C_w", "C_e", "C0"] if use_E else ["C_d", "C_w", "C0"])
    return dict(zip(keys, coef))

def predicted_T_step(coef, D, W, E, use_E):
    if use_E:
        return coef["C_d"]*D + coef["C_w"]*W + coef["C_e"]*E + coef["C0"]
    return coef["C_d"]*D + coef["C_w"]*W + coef["C0"]

def E_lookup(shapes_at_lp, D, W):
    if (D, W) in shapes_at_lp:
        return shapes_at_lp[(D, W)]["mean_distinct_experts"]
    Ws_avail = sorted({w for (d, w) in shapes_at_lp if d == D})
    if len(Ws_avail) >= 2 and W not in Ws_avail:
        lo = max([w for w in Ws_avail if w < W], default=None)
        hi = min([w for w in Ws_avail if w > W], default=None)
        if lo and hi:
            e_lo = shapes_at_lp[(D, lo)]["mean_distinct_experts"]
            e_hi = shapes_at_lp[(D, hi)]["mean_distinct_experts"]
            frac = (math.log2(W) - math.log2(lo)) / (math.log2(hi) - math.log2(lo))
            return e_lo + frac * (e_hi - e_lo)
    raise KeyError(f"no E for D={D} W={W}")

def acceptance_fn(acc_params, acc_lookup, D, W):
    if acc_params:
        L, r, g = acc_params["L"], acc_params["r"], acc_params["g"]
        return L * (1 - r**D) * (1 + g * math.log2(W))
    return acc_lookup.get((D, W), acc_lookup.get((D, 1), 1.0))

def greedy_pick(shapes_at_lp, coef, use_E, acc_params, acc_lookup):
    valid_dw = sorted(shapes_at_lp.keys())
    valid_D = sorted({d for d, w in valid_dw})
    valid_W = sorted({w for d, w in valid_dw})
    D_idx, W_idx = 0, 0
    D, W = valid_D[D_idx], valid_W[W_idx]

    def Tstep(D, W):
        E = E_lookup(shapes_at_lp, D, W) if use_E else None
        return predicted_T_step(coef, D, W, E, use_E)

    while True:
        ell = acceptance_fn(acc_params, acc_lookup, D, W)
        T = Tstep(D, W)
        avg = ell / T if T > 0 else float("inf")
        can_D = D_idx + 1 < len(valid_D)
        can_W = W_idx + 1 < len(valid_W)
        if not can_D and not can_W:
            break
        m_D = -float("inf")
        if can_D:
            Dn = valid_D[D_idx+1]
            elln, Tn = acceptance_fn(acc_params, acc_lookup, Dn, W), Tstep(Dn, W)
            m_D = (elln - ell) / (Tn - T) if Tn != T else float("inf")
        m_W = -float("inf")
        if can_W:
            Wn = valid_W[W_idx+1]
            elln, Tn = acceptance_fn(acc_params, acc_lookup, D, Wn), Tstep(D, Wn)
            m_W = (elln - ell) / (Tn - T) if Tn != T else float("inf")
        best_m = max(m_D, m_W)
        if best_m <= avg:
            break
        if m_D >= m_W:
            D_idx += 1
        else:
            W_idx += 1
        D, W = valid_D[D_idx], valid_W[W_idx]
    return D, W

def main():
    grid = json.load(open(GRID_PATH))
    all_cells = [v for v in grid.values() if "error" not in v]
    print(f"Loaded {len(all_cells)} valid cells")

    by_load = {}
    for c in all_cells:
        by_load.setdefault((c["B"], c["rate"]), {})[(c["num_steps"], c["eagle_topk"])] = c
    load_points = sorted(by_load.keys())

    acc_params = fit_acceptance(all_cells)
    acc_lookup = acceptance_lookup(all_cells)
    print(f"Acceptance model: {acc_params if acc_params else '(using raw lookup, fit failed)'}")

    all_shapes = set.intersection(*[set(s.keys()) for s in by_load.values()])
    b0_scores = {s: np.mean([goodput(by_load[lp][s]) for lp in load_points]) for s in all_shapes}
    b0_shape = max(b0_scores, key=b0_scores.get)
    print(f"B0 (fixed default, best-average over {len(all_shapes)} common shapes): D{b0_shape[0]}W{b0_shape[1]}")

    results = {}
    for lp in load_points:
        shapes = by_load[lp]
        best_shape, best_cell = max(shapes.items(), key=lambda kv: goodput(kv[1]))
        b0_cell = shapes.get(b0_shape)
        b0_gp = goodput(b0_cell) if b0_cell else None
        coef_b1 = fit_cost_model(shapes, use_E=False)
        d1, w1 = greedy_pick(shapes, coef_b1, False, acc_params, acc_lookup)
        b1_cell = shapes.get((d1, w1))
        b1_gp = goodput(b1_cell) if b1_cell else None
        coef_b2 = fit_cost_model(shapes, use_E=True)
        d2, w2 = greedy_pick(shapes, coef_b2, True, acc_params, acc_lookup)
        b2_cell = shapes.get((d2, w2))
        b2_gp = goodput(b2_cell) if b2_cell else None

        key = f"B{lp[0]}_r{lp[1]:g}"
        results[key] = {
            "B": lp[0], "rate": lp[1],
            "B0_shape": f"D{b0_shape[0]}W{b0_shape[1]}", "B0_goodput": b0_gp,
            "B1_shape": f"D{d1}W{w1}", "B1_goodput": b1_gp,
            "B2_shape": f"D{d2}W{w2}", "B2_goodput": b2_gp,
            "oracle_shape": f"D{best_shape[0]}W{best_shape[1]}", "oracle_goodput": goodput(best_cell),
        }

    print(f"\n{'load':>10} {'B0':>14} {'B1':>14} {'B2':>14} {'oracle':>14} {'B2vB0':>8} {'B2vB1':>8}")
    for k, v in results.items():
        b2vb0 = (v['B2_goodput']/v['B0_goodput']-1)*100 if v['B0_goodput'] else float('nan')
        b2vb1 = (v['B2_goodput']/v['B1_goodput']-1)*100 if v['B1_goodput'] else float('nan')
        print(f"{k:>10} {v['B0_shape']:>5}{v['B0_goodput']:>8.1f} "
              f"{v['B1_shape']:>5}{v['B1_goodput']:>8.1f} "
              f"{v['B2_shape']:>5}{v['B2_goodput']:>8.1f} "
              f"{v['oracle_shape']:>5}{v['oracle_goodput']:>8.1f} "
              f"{b2vb0:>7.1f}% {b2vb1:>7.1f}%")

    json.dump({"acceptance_model": acc_params, "B0_shape": f"D{b0_shape[0]}W{b0_shape[1]}",
               "load_points": results}, open(OUT_PATH, "w"), indent=2)
    print(f"\nWrote {OUT_PATH}")

if __name__ == "__main__":
    main()
