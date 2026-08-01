"""Axis-4: does a TTFT-predictive admission sensor beat the ratio sensor?

Follow-up to scripts/sweep_admit_kv.py (Axis-3). That sweep's adaptive arms
tied the static baseline everywhere, and the step traces say why:

  1. The admission sensor cannot respond proportionally to a TTFT breach.
     TTFTSlackAdmit/KVAwareAdmit compare num_waiting/cap against
     target_util=0.7, so slack = 0.7 - waiting/cap is bounded above by 0.7 and
     is small in magnitude for any queue shorter than the cap. Two regimes,
     both observed:
       * reason/ngram rate=2: waiting peaked at 37 vs cap=256 (ratio 0.14), so
         the law saw pure slack and the cap sat at 256 all run -- while TTFT
         p99 was 37s against a 2.0s SLO, a 19x breach.
       * code/EAGLE rate=6: waiting peaked at 207 vs cap=256 (ratio 0.81), so
         the law DID cross its threshold -- and still only moved the cap from
         256 to 250, because slack=-0.109 gives delta = 0.5*8*(-0.109) = -0.43
         per update against a 256-wide cap.
     At that same step the predictive law computes a 138.6s predicted wait
     against a 1.4s budget, slack=-98, delta=-392 (clamped to the floor) --
     same gain, same actuation code, ~900x the response. The defect is the
     sensor's scaling, not merely its threshold.
  2. gamma had no leverage on the ngram substrate: TPOT p50 was identical to
     three decimals across gamma=1/2/4 at every rate.
  3. Every cell at rate>=1.5 was in open-loop overload (e2e p95 ~= the whole
     60s run), where all admission policies converge by construction.

This sweep changes all three: the predictive sensor (controllers.
_PredictedWaitTerm), EAGLE speculation, rates concentrated at the SLO knee
(attainment fell 0.93 -> 0.35 between rate=1.0 and 1.5), 300s runs so the
queue reaches steady state, and multiple seeds so a few-point gap can be
called.

Arms are paired so each comparison isolates exactly one change:

  static-cap-hi       fixed cap=256, fixed gamma=4. The Axis-1/2/3 baseline.
  ratio-cap           TTFTSlackAdmit: the OLD ratio sensor. Present so
                      "predictive beats ratio" is measured, not assumed.
  predictive-cap      TTFTPredictiveAdmit: the NEW sensor, same actuation.
                      predictive-cap minus ratio-cap IS the sensor fix.
  predictive-kv       + the proportional KV ceiling on top.
  predictive-full     + gated gamma. Only differs from predictive-kv by gamma.
  predictive-shed     predictive-kv + admission-time shedding. Scored on
                      slo_attainment_offered (shed counted as failures).
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

STATIC_CAP_HI = 256
GAMMA_STATIC = 4
GAMMA_FLOOR = 1
# Mean output length per workload -- the one workload-specific constant the
# predictive sensor needs (see controllers._PredictedWaitTerm). These are
# measured mean_output_tokens from real runs, not trace max_tokens: code=316
# from the EAGLE smoke run, reason=840 from the Axis-3 grids. rag/chat are
# still trace-derived estimates and should be re-measured before use.
# Override with --est-output-len for anything not listed.
EST_OUTPUT_LEN_BY_RTYPE = {"reason": 840.0, "code": 316.0, "rag": 116.0, "chat": 470.0}
# threshold re-derived for the EAGLE/code pairing, as GatedSpec.__init__'s own
# comment requires ("re-derive from mean_accept_rate on whatever model/draft
# method this runs against next"). Measured acceptance on this pairing spans
# 0.34 (rate=8, at the knee) to 0.44 (light load). The setpoint law
# target = log(threshold)/log(accept) then gives:
#     threshold=0.15 -> gamma 1.76..2.31   (always below the gamma=4 baseline)
#     threshold=0.05 -> gamma 2.78..3.65   (straddles it)
# 0.05 is used so the gated arm is not structurally pinned under the static
# baseline: it can land on either side depending on measured acceptance, which
# is what makes the comparison a test rather than a foregone conclusion.
_GATED_KW = {"threshold": 0.05, "gain": 0.5, "deadband": 0.5, "period": 4,
             "gamma_min": 0, "gamma_max": 8, "gamma_init": GAMMA_STATIC,
             "gamma_floor": GAMMA_FLOOR, "decode_bound_util": 0.85,
             "kv_headroom_frac": 0.15}
_KV_KW = {"kv_target": 0.80, "kv_min_frac": 0.30}

ARM_NAMES = ["static-cap-hi", "ratio-cap", "predictive-cap", "predictive-kv",
             "predictive-full", "predictive-shed"]


def build_arm(arm: str, est_len: float) -> dict:
    """Arm spec for one workload. ``est_len`` is the predictive sensor's mean
    output-length estimate and is workload-specific, so arms are built per run
    rather than being a module-level constant."""
    pred_kw = {"ttft_target_util": 0.7, "est_output_len": est_len,
               "gain": 0.5, "period": 4, "batch_min": 1,
               "batch_max": STATIC_CAP_HI, "init": STATIC_CAP_HI}
    if arm == "static-cap-hi":
        return {"controller": {"spec": "static", "admit": "static",
                               "coordination": "naive",
                               "spec_kw": {"gamma": GAMMA_STATIC},
                               "admit_kw": {"max_num_seqs": STATIC_CAP_HI}}}
    if arm == "ratio-cap":
        return {"controller": {"spec": "static", "admit": "ttft-slack",
                               "coordination": "naive",
                               "spec_kw": {"gamma": GAMMA_FLOOR},
                               "admit_kw": {"target_util": 0.7, "gain": 0.5,
                                            "period": 4, "batch_min": 1,
                                            "batch_max": STATIC_CAP_HI,
                                            "init": STATIC_CAP_HI}}}
    if arm == "predictive-cap":
        return {"controller": {"spec": "static", "admit": "ttft-predictive",
                               "coordination": "naive",
                               "spec_kw": {"gamma": GAMMA_FLOOR},
                               "admit_kw": dict(pred_kw)}}
    if arm == "predictive-kv":
        return {"controller": {"spec": "static", "admit": "kv-predictive",
                               "coordination": "naive",
                               "spec_kw": {"gamma": GAMMA_FLOOR},
                               "admit_kw": {**pred_kw, **_KV_KW}}}
    if arm == "predictive-full":
        return {"controller": {"spec": "gated", "admit": "kv-predictive",
                               "coordination": "naive",
                               "spec_kw": dict(_GATED_KW),
                               "admit_kw": {**pred_kw, **_KV_KW}}}
    if arm == "predictive-shed":
        return {"controller": {"spec": "gated", "admit": "kv-predictive",
                               "coordination": "naive",
                               "spec_kw": dict(_GATED_KW),
                               "admit_kw": {**pred_kw, **_KV_KW}},
                "runtime": {"shed_enabled": True, "shed_slo_mult": 1.0,
                            "shed_est_output_len": est_len}}
    raise KeyError(arm)


def run_cell(base_cfg, arm, rtype, rate, duration, seed, out_dir, est_len):
    cfg = copy.deepcopy(base_cfg)
    spec = build_arm(arm, est_len)
    cfg["runtime"]["max_num_seqs_init"] = STATIC_CAP_HI
    cfg["runtime"].update(spec.get("runtime", {}))
    cfg["controller"] = copy.deepcopy(spec["controller"])
    cfg_path = os.path.join(out_dir, f"cfg_{arm}_{rtype}_r{rate}_s{seed}.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    run_dir = os.path.join(out_dir, f"{arm}_{rtype}_r{rate}_s{seed}")
    cmd = [sys.executable, "-m", "specloop_rt.replay", "--config", cfg_path,
           "--trace", "homogeneous", "--rtype", rtype, "--rate", str(rate),
           "--duration", str(duration), "--seed", str(seed),
           "--real-corpus", "--out", run_dir]
    print(f">> arm={arm} rtype={rtype} rate={rate} seed={seed}", flush=True)
    subprocess.run(cmd, check=True)
    return summarize_run(run_dir, tpot_slo=base_cfg["runtime"]["tpot_slo_s"],
                         ttft_slo=base_cfg["runtime"]["ttft_slo_s"])


def main(argv=None):
    p = argparse.ArgumentParser("axis-4 predictive-admission sweep")
    p.add_argument("--config", required=True)
    p.add_argument("--rtypes", nargs="+", default=["code"])
    p.add_argument("--rates", type=float, nargs="+", default=[1.0, 1.25, 1.5])
    p.add_argument("--arms", nargs="+", default=list(ARM_NAMES))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--duration", type=float, default=300.0)
    p.add_argument("--est-output-len", type=float, default=None,
                   help="predictive sensor's mean output-length estimate; "
                        "defaults to the per-rtype measured value")
    p.add_argument("--out", default="results_gpu_sweep/axis4")
    a = p.parse_args(argv)

    for arm in a.arms:
        if arm not in ARM_NAMES:
            raise SystemExit(f"unknown arm {arm!r}; choices: {ARM_NAMES}")

    os.makedirs(a.out, exist_ok=True)
    grid_path = os.path.join(a.out, "grid.json")
    grid = json.load(open(grid_path)) if os.path.exists(grid_path) else {}

    with open(a.config) as f:
        base_cfg = yaml.safe_load(f)

    for arm in a.arms:
        for rtype in a.rtypes:
            for rate in a.rates:
                for seed in a.seeds:
                    key = f"{arm}_{rtype}_r{rate}_s{seed}"
                    if key in grid and "error" not in grid[key]:
                        print(f".. skip {key}", flush=True)
                        continue
                    est_len = (a.est_output_len if a.est_output_len is not None
                               else EST_OUTPUT_LEN_BY_RTYPE.get(rtype, 400.0))
                    try:
                        m = run_cell(base_cfg, arm, rtype, rate, a.duration,
                                     seed, a.out, est_len)
                        grid[key] = {"arm": arm, "rtype": rtype, "rate": rate,
                                     "seed": seed, **m}
                    except subprocess.CalledProcessError as e:
                        print(f"!! {key} FAILED: {e}", flush=True)
                        grid[key] = {"arm": arm, "rtype": rtype, "rate": rate,
                                     "seed": seed, "error": str(e)}
                    with open(grid_path, "w") as f:
                        json.dump(grid, f, indent=2)

    # ---- seed-averaged summary, with spread so ties are visible as ties ----
    import statistics as st
    # Column widths account for the "mean±sd" suffix, not just the mean: an
    # earlier version reserved 16 chars for a string like "0.090±0.070" plus a
    # neighbour of "30.099±8.397" and the two ran together unreadably.
    W = 17
    print("\n=== seed-averaged per arm x rate (mean±sd over seeds) ===")
    print("arm".ljust(18) + "rate".rjust(6) + "slo".rjust(W) +
          "slo_offered".rjust(W) + "ttft_p99".rjust(W) + "gamma".rjust(W) +
          "shed".rjust(W))
    for arm in a.arms:
        for rate in a.rates:
            rows = [v for v in grid.values()
                    if v.get("arm") == arm and v.get("rate") == rate and "error" not in v]
            if not rows:
                continue
            def agg(k, prec=3):
                vals = [r[k] for r in rows if k in r and r[k] == r[k]]
                if not vals:
                    return "n/a"
                mu = sum(vals) / len(vals)
                sd = st.stdev(vals) if len(vals) > 1 else 0.0
                return f"{mu:.{prec}f}±{sd:.{prec}f}"
            print(arm.ljust(18) + f"{rate:>6}" + agg("slo_attainment").rjust(W) +
                  agg("slo_attainment_offered").rjust(W) +
                  agg("ttft_p99", 1).rjust(W) + agg("mean_gamma", 2).rjust(W) +
                  agg("shed_frac", 2).rjust(W))

    n_failed = sum(1 for v in grid.values() if "error" in v)
    print(f"\n{len(grid) - n_failed}/{len(grid)} cells succeeded -> {grid_path}")
    print("\nRead order: predictive-cap minus ratio-cap = the sensor fix alone "
          "(the headline number). predictive-kv minus predictive-cap = KV "
          "ceiling on a cap loop that actually actuates. predictive-full minus "
          "predictive-kv = gated gamma on the EAGLE substrate. predictive-shed "
          "must be read on slo_attainment_offered, never slo_attainment -- "
          "shedding trivially inflates the admitted-set number.")


if __name__ == "__main__":
    main()
