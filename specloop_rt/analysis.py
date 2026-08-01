"""Analysis over real-run telemetry.

Consumes the two artifacts a run produces:
  steps.jsonl     — one record per scheduler step (control + system state)
  requests.jsonl  — one record per request (TTFT/TPOT/E2E)

Produces the same metric families as the simulator study so GPU results are
directly comparable, plus the control-stability signatures (spectral
concentration, CV, settling time, cross-correlation) that distinguish a
limit-cycling loop from a stable one.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def load_run(run_dir: str) -> Dict[str, pd.DataFrame]:
    steps, meta = [], {}
    with open(f"{run_dir}/steps.jsonl") as f:
        for line in f:
            o = json.loads(line)
            if "_meta" in o:
                meta = o["_meta"]; continue
            steps.append(o)
    reqs = [json.loads(l) for l in open(f"{run_dir}/requests.jsonl")]
    return {"steps": pd.DataFrame(steps), "requests": pd.DataFrame(reqs), "meta": meta}


# ---- end metrics ----------------------------------------------------------

def end_metrics(run: Dict, tpot_slo: float, ttft_slo: float) -> Dict[str, float]:
    r = run["requests"]
    s = run["steps"]
    done = r[r.finish_wall.notna()]
    if done.empty:
        return {"n_finished": 0}
    ttft = done.ttft_s.dropna().values
    tpot = done.tpot_s.dropna().values
    e2e = done.e2e_s.dropna().values
    slo_ok = ((done.ttft_s <= ttft_slo) & (done.tpot_s <= tpot_slo)).mean()
    ttft_ok = (done.ttft_s <= ttft_slo).mean()
    tpot_ok = (done.tpot_s <= tpot_slo).mean()
    # Load shedding accounting: a shed request produced no tokens and met no
    # SLO, so it must count against attainment. Reporting only the admitted
    # set would make "reject almost everything" the trivially optimal policy.
    # slo_attainment stays the admitted-set number (comparable to every prior
    # round, none of which shed); *_offered are over all arrivals.
    n_shed = int(r["shed"].sum()) if "shed" in r.columns else 0
    n_offered = len(r)
    slo_ok_offered = (float(slo_ok) * len(done) / n_offered) if n_offered else float("nan")
    dur = s.t_wall.max() - s.t_wall.min() if len(s) else 1.0
    out_tok = done.output_tokens.sum()
    mean_out_tok = float(done.output_tokens.mean())
    ttft_p99 = float(np.percentile(ttft, 99))
    tpot_p99 = float(np.percentile(tpot, 99))
    # bound-edness: compare each P99 directly to its own SLO rather than a
    # ratio of the two (a ratio can be fooled by output length -- a genuine
    # queueing blowup, e.g. ttft_p99=13s at rate=4/len=1024 with slo_attainment
    # 0.55, still came out "decode_bound" under P99_ttft/(P99_tpot*mean_len)
    # because dividing by 1024 washed out the numerator; confirmed on real
    # GPU data in the axis-1 sweep before switching to this formulation).
    # "breach" = this SLO alone would fail attainment at this load.
    ttft_breach = ttft_p99 > ttft_slo
    tpot_breach = tpot_p99 > tpot_slo
    if ttft_breach and tpot_breach:
        regime = "both_bound"
    elif ttft_breach:
        regime = "admission_bound"
    elif tpot_breach:
        regime = "decode_bound"
    else:
        regime = "unconstrained"
    return {
        "n_finished": len(done),
        "ttft_p50": float(np.percentile(ttft, 50)),
        "ttft_p95": float(np.percentile(ttft, 95)), "ttft_p99": ttft_p99,
        # tpot_s is the per-request mean inter-token latency (ITL averaged
        # over that request's decode steps) -- this IS the ITL metric; no
        # separate field, to avoid two names for the same quantity.
        "tpot_p50": float(np.percentile(tpot, 50)), "tpot_p99": tpot_p99,
        "e2e_p50": float(np.percentile(e2e, 50)),
        "e2e_p95": float(np.percentile(e2e, 95)), "e2e_p99": float(np.percentile(e2e, 99)),
        "slo_attainment": float(slo_ok),
        # joint AND above collapses to ~0 whenever one SLO is structurally
        # unreachable at a given load (e.g. TTFT under admission queueing) --
        # these split it out so a single tight constraint doesn't mask that
        # the other constraint is being met fine.
        "ttft_attainment": float(ttft_ok),
        "tpot_attainment": float(tpot_ok),
        "n_shed": n_shed,
        "n_offered": n_offered,
        "shed_frac": float(n_shed / n_offered) if n_offered else float("nan"),
        # attainment over OFFERED load (shed requests counted as failures) --
        # the number to compare when any arm sheds; equals slo_attainment when
        # nothing is shed and every arrival finished.
        "slo_attainment_offered": float(slo_ok_offered),
        "goodput_tok_s": float(out_tok / dur),
        "mean_gamma": float(s.act_gamma.dropna().mean()) if "act_gamma" in s else float("nan"),
        "mean_accept_rate": float(s.accept_rate_ema.mean()),
        "mean_running": float(s.num_running.mean()),
        "mean_kv_frac": float((s.kv_used_blocks / s.kv_total_blocks.clip(lower=1)).mean()),
        "mean_output_tokens": mean_out_tok,
        "ttft_over_slo": float(ttft_p99 / ttft_slo) if ttft_slo > 0 else float("inf"),
        "tpot_over_slo": float(tpot_p99 / tpot_slo) if tpot_slo > 0 else float("inf"),
        "regime": regime,
    }


# ---- control-stability metrics (shared with the simulator's metrics.py) ---

def _detrend(x):
    x = np.asarray(x, float)
    if x.size < 4:
        return x - x.mean() if x.size else x
    t = np.arange(x.size)
    A = np.vstack([t, np.ones_like(t)]).T
    c, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ c


def oscillation(series, dt_mean, prefix=""):
    x = np.asarray(series, float)
    out = {f"{prefix}cv": 0.0, f"{prefix}dom_freq_hz": 0.0,
           f"{prefix}spectral_concentration": 0.0}
    if x.size < 32:
        return out
    mu = x.mean()
    out[f"{prefix}cv"] = float(x.std() / mu) if mu > 1e-9 else 0.0
    xd = _detrend(x) * np.hanning(x.size)
    spec = np.abs(np.fft.rfft(xd)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=max(dt_mean, 1e-9))
    if spec.size > 2:
        spec[0] = 0.0
        tot = spec.sum(); k = int(np.argmax(spec))
        out[f"{prefix}dom_freq_hz"] = float(freqs[k])
        out[f"{prefix}spectral_concentration"] = float(spec[k] / tot) if tot > 0 else 0.0
    return out


def settling(series, t, t_event, tol=0.05, prefix=""):
    x = np.asarray(series, float); t = np.asarray(t, float)
    out = {f"{prefix}overshoot": 0.0, f"{prefix}settling_s": float("nan")}
    pre, post, tpost = x[t < t_event], x[t >= t_event], t[t >= t_event]
    if pre.size < 8 or post.size < 32:
        return out
    tail = post[int(0.7 * post.size):]
    final = float(tail.mean()) if tail.size else float(post.mean())
    span = max(abs(final - pre.mean()), 1e-9)
    out[f"{prefix}overshoot"] = float(np.max(np.abs(post - final)) / span)
    band = tol * max(abs(final), 1e-9)
    idx = np.where(np.abs(post - final) > band)[0]
    out[f"{prefix}settling_s"] = (0.0 if idx.size == 0
                                  else float(tpost[min(idx[-1] + 1, post.size - 1)] - t_event))
    return out


def stability_metrics(run: Dict, t_event: Optional[float] = None, warmup_frac=0.1) -> Dict:
    s = run["steps"]
    if s.empty:
        return {}
    s = s.iloc[int(warmup_frac * len(s)):]
    t = s.t_wall.values - run["steps"].t_wall.min()
    dt = float(np.mean(np.diff(t))) if len(t) > 1 else 0.01
    out = {}
    out.update(oscillation(s.act_gamma.ffill().values, dt, "gamma_"))
    out.update(oscillation(s.num_running.values, dt, "batch_"))
    g = _detrend(s.act_gamma.ffill().values)
    b = _detrend(s.num_running.values.astype(float))
    if g.std() > 1e-9 and b.std() > 1e-9:
        out["gamma_batch_corr"] = float(np.corrcoef(g, b)[0, 1])
    else:
        out["gamma_batch_corr"] = 0.0
    if t_event is not None:
        out.update(settling(s.act_gamma.ffill().values, t, t_event, prefix="gamma_"))
    # acceptance-signal noise: the input the closed-loop controller reacts to,
    # not just gamma (its output). A high accept_rate_cv with a low-magnitude
    # gamma response would still mean the controller is being driven by a
    # noisy signal -- this is the number the sensing-noise sub-experiment
    # sweeps against (EMA window / min-sample gate vs. this CV).
    acc = s.accept_rate_ema.dropna().values
    out["accept_rate_mean"] = float(acc.mean()) if acc.size else float("nan")
    out["accept_rate_cv"] = float(acc.std() / acc.mean()) if acc.size and acc.mean() > 1e-9 else 0.0
    return out


def is_oscillatory(m: Dict, conc=0.10, cv=0.12) -> bool:
    return m.get("gamma_spectral_concentration", 0) > conc and m.get("gamma_cv", 0) > cv


def summarize_run(run_dir: str, tpot_slo: float, ttft_slo: float,
                  t_event: Optional[float] = None) -> Dict:
    run = load_run(run_dir)
    m = {"run_dir": run_dir}
    m.update(end_metrics(run, tpot_slo, ttft_slo))
    m.update(stability_metrics(run, t_event))
    m["oscillatory"] = is_oscillatory(m)
    return m
