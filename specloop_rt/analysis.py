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
    dur = s.t_wall.max() - s.t_wall.min() if len(s) else 1.0
    out_tok = done.output_tokens.sum()
    return {
        "n_finished": len(done),
        "ttft_p50": float(np.percentile(ttft, 50)), "ttft_p99": float(np.percentile(ttft, 99)),
        "tpot_p50": float(np.percentile(tpot, 50)), "tpot_p99": float(np.percentile(tpot, 99)),
        "e2e_p95": float(np.percentile(e2e, 95)),
        "slo_attainment": float(slo_ok),
        "goodput_tok_s": float(out_tok / dur),
        "mean_gamma": float(s.act_gamma.dropna().mean()) if "act_gamma" in s else float("nan"),
        "mean_accept_rate": float(s.accept_rate_ema.mean()),
        "mean_running": float(s.num_running.mean()),
        "mean_kv_frac": float((s.kv_used_blocks / s.kv_total_blocks.clip(lower=1)).mean()),
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
    out.update(oscillation(s.act_gamma.fillna(method="ffill").values, dt, "gamma_"))
    out.update(oscillation(s.num_running.values, dt, "batch_"))
    g = _detrend(s.act_gamma.fillna(method="ffill").values)
    b = _detrend(s.num_running.values.astype(float))
    if g.std() > 1e-9 and b.std() > 1e-9:
        out["gamma_batch_corr"] = float(np.corrcoef(g, b)[0, 1])
    else:
        out["gamma_batch_corr"] = 0.0
    if t_event is not None:
        out.update(settling(s.act_gamma.fillna(method="ffill").values, t, t_event, prefix="gamma_"))
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
