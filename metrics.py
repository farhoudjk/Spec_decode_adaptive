"""End-metrics and control-stability metrics.

The stability metrics are the load-bearing ones for RQ1/RQ2: an interference
claim must be evidenced by signatures that are *absent* in the isolated-loop
runs, not merely by worse averages.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ==========================================================================
# End metrics
# ==========================================================================


def end_metrics(result) -> Dict[str, float]:
    fin = result.finished
    cfg = result.cfg
    if not fin:
        return {"n_finished": 0}

    ttft = np.array([r.first_token_s - r.arrival_s for r in fin if r.first_token_s])
    tpot = np.array([np.mean(r.itls) for r in fin if r.itls])
    e2e = np.array([r.finished_s - r.arrival_s for r in fin if r.finished_s])
    out_tok = sum(r.generated for r in fin)

    slo_ok = np.array([
        (r.first_token_s - r.arrival_s <= cfg.ttft_slo_s) and
        (np.mean(r.itls) <= cfg.tpot_slo_s)
        for r in fin if r.itls and r.first_token_s
    ])

    df = result.to_frame()
    busy_s = float(df.loc[df.batch_size > 0, "step_time"].sum())
    gpu_s = busy_s * result.hardware.num_gpus

    return {
        "n_submitted": result.n_submitted,
        "n_finished": len(fin),
        "n_dropped": result.dropped,
        "wall_s": result.wall_s,
        "ttft_p50": float(np.percentile(ttft, 50)),
        "ttft_p95": float(np.percentile(ttft, 95)),
        "ttft_p99": float(np.percentile(ttft, 99)),
        "tpot_p50": float(np.percentile(tpot, 50)),
        "tpot_p95": float(np.percentile(tpot, 95)),
        "tpot_p99": float(np.percentile(tpot, 99)),
        "e2e_p95": float(np.percentile(e2e, 95)),
        "slo_attainment": float(slo_ok.mean()) if slo_ok.size else 0.0,
        "goodput_tok_s": out_tok / result.wall_s if result.wall_s > 0 else 0.0,
        "gpu_seconds": gpu_s,
        "gpu_s_per_ktok": 1000.0 * gpu_s / max(1, out_tok),
        "mean_gamma": float(df.gamma.mean()),
        "mean_batch": float(df.batch_size.mean()),
        "mean_accepted": float(df.accepted_mean.mean()),
        "preemptions": int(df.preempted.sum()),
    }


# ==========================================================================
# Control-stability metrics
# ==========================================================================


def _detrend(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size < 4:
        return x - x.mean() if x.size else x
    t = np.arange(x.size)
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ coef


def oscillation_metrics(series: np.ndarray, dt_mean: float,
                        prefix: str = "") -> Dict[str, float]:
    """Frequency-domain signature of limit-cycling.

    ``spectral_concentration`` is the share of detrended power in the single
    dominant non-DC bin: white noise -> ~0, a limit cycle -> large.  This is the
    statistic to threshold when declaring a run "oscillatory".
    """
    x = np.asarray(series, dtype=float)
    out = {f"{prefix}cv": 0.0, f"{prefix}dom_freq_hz": 0.0,
           f"{prefix}spectral_concentration": 0.0, f"{prefix}n_reversals": 0.0}
    if x.size < 32:
        return out
    mu = np.mean(x)
    out[f"{prefix}cv"] = float(np.std(x) / mu) if mu > 1e-9 else 0.0

    xd = _detrend(x)
    xd = xd * np.hanning(xd.size)
    spec = np.abs(np.fft.rfft(xd)) ** 2
    freqs = np.fft.rfftfreq(xd.size, d=max(dt_mean, 1e-9))
    if spec.size > 2:
        spec[0] = 0.0
        total = spec.sum()
        k = int(np.argmax(spec))
        out[f"{prefix}dom_freq_hz"] = float(freqs[k])
        out[f"{prefix}spectral_concentration"] = float(spec[k] / total) if total > 0 else 0.0

    d = np.diff(x)
    s = np.sign(d)
    s = s[s != 0]
    if s.size > 1:
        out[f"{prefix}n_reversals"] = float(np.sum(np.diff(s) != 0) / s.size)
    return out


def settling_metrics(series: np.ndarray, t: np.ndarray, t_event: float,
                     tol: float = 0.05, prefix: str = "") -> Dict[str, float]:
    """Overshoot and settling time after a step disturbance at ``t_event``."""
    x = np.asarray(series, dtype=float)
    t = np.asarray(t, dtype=float)
    out = {f"{prefix}overshoot": 0.0, f"{prefix}settling_s": float("nan"),
           f"{prefix}pre_mean": 0.0, f"{prefix}post_mean": 0.0}
    pre = x[t < t_event]
    post = x[t >= t_event]
    tpost = t[t >= t_event]
    if pre.size < 8 or post.size < 32:
        return out
    out[f"{prefix}pre_mean"] = float(pre.mean())

    tail = post[int(0.7 * post.size):]
    final = float(tail.mean()) if tail.size else float(post.mean())
    out[f"{prefix}post_mean"] = final

    span = max(abs(final - pre.mean()), 1e-9)
    out[f"{prefix}overshoot"] = float((np.max(np.abs(post - final)) ) / span)

    band = tol * max(abs(final), 1e-9)
    inside = np.abs(post - final) <= band
    # last index where it leaves the band
    idx = np.where(~inside)[0]
    if idx.size == 0:
        out[f"{prefix}settling_s"] = 0.0
    elif idx[-1] + 1 < post.size:
        out[f"{prefix}settling_s"] = float(tpost[idx[-1] + 1] - t_event)
    else:
        out[f"{prefix}settling_s"] = float(tpost[-1] - t_event)   # never settles
    return out


def stability_metrics(result, t_event: Optional[float] = None,
                      warmup_frac: float = 0.1) -> Dict[str, float]:
    df = result.to_frame()
    if df.empty:
        return {}
    n0 = int(warmup_frac * len(df))
    d = df.iloc[n0:]
    dt = float(d.step_time.mean())

    out: Dict[str, float] = {}
    out.update(oscillation_metrics(d.gamma.values, dt, prefix="gamma_"))
    out.update(oscillation_metrics(d.batch_size.values, dt, prefix="batch_"))
    out.update(oscillation_metrics(d.max_batch.values, dt, prefix="cap_"))

    # cross-loop coupling: correlation between the two actuation signals
    if d.gamma.std() > 1e-9 and d.max_batch.std() > 1e-9:
        out["gamma_cap_corr"] = float(np.corrcoef(_detrend(d.gamma.values),
                                                  _detrend(d.max_batch.values))[0, 1])
        # lagged cross-correlation peak: evidence one loop is chasing the other
        a = _detrend(d.gamma.values)
        b = _detrend(d.max_batch.values)
        a = (a - a.mean()) / (a.std() + 1e-12)
        b = (b - b.mean()) / (b.std() + 1e-12)
        n = min(a.size, 4000)
        xcorr = np.correlate(a[:n], b[:n], mode="full") / n
        lags = np.arange(-n + 1, n)
        keep = np.abs(lags) <= 200
        out["xcorr_peak"] = float(np.max(np.abs(xcorr[keep])))
        out["xcorr_lag"] = float(lags[keep][int(np.argmax(np.abs(xcorr[keep])))])
    else:
        out["gamma_cap_corr"] = 0.0
        out["xcorr_peak"] = 0.0
        out["xcorr_lag"] = 0.0

    if t_event is not None:
        out.update(settling_metrics(df.gamma.values, df.t.values, t_event, prefix="gamma_"))
        out.update(settling_metrics(df.batch_size.values, df.t.values, t_event, prefix="batch_"))
    return out


def is_oscillatory(row: Dict[str, float], conc_thresh: float = 0.10,
                   cv_thresh: float = 0.12) -> bool:
    """Binary classification used to draw the RQ2 stability map."""
    return (row.get("gamma_spectral_concentration", 0) > conc_thresh and
            row.get("gamma_cv", 0) > cv_thresh)


def summarize(records: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(records)


def bootstrap_ci(x, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(bs, 100 * alpha / 2)), float(np.percentile(bs, 100 * (1 - alpha / 2)))
