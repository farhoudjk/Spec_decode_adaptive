"""Figures and LaTeX tables.

Figure style follows a two-column camera-ready: serif family, large fonts,
colorblind-safe palette with distinct markers.  The RQ2 stability map is
emitted as a booktabs + cellcolor table rather than a raster heatmap.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Okabe-Ito, colorblind-safe
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 15,
        "axes.labelsize": 17,
        "axes.titlesize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


# ==========================================================================
# Figures
# ==========================================================================


def fig_rq1_timeseries(traces: Dict[str, pd.DataFrame], t_event: float, path: str):
    """gamma and batch-size trajectories for the four isolation arms."""
    set_style()
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)
    for i, (label, d) in enumerate(traces.items()):
        c, m = PALETTE[i % len(PALETTE)], MARKERS[i % len(MARKERS)]
        axes[0].plot(d.t, d.gamma, color=c, label=label, marker=m,
                     markevery=max(1, len(d) // 25))
        axes[1].plot(d.t, d.batch_size, color=c, marker=m,
                     markevery=max(1, len(d) // 25))
    for ax in axes:
        ax.axvline(t_event, color="0.35", ls="--", lw=1.5)
    axes[0].set_ylabel(r"speculation length $\gamma$")
    axes[1].set_ylabel("batch size")
    axes[1].set_xlabel("time (s)")
    axes[0].legend(ncol=2, frameon=False)
    fig.savefig(path)
    plt.close(fig)


def fig_rq1_bars(df: pd.DataFrame, path: str,
                 metrics=("gamma_spectral_concentration", "gamma_cv", "gamma_settling_s")):
    set_style()
    arms = ["A_none", "B_spec_only", "C_admit_only", "D_both_naive"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.4))
    axes = np.atleast_1d(axes)
    for ax, met in zip(axes, metrics):
        mu = [df[df.arm == a][met].mean() for a in arms]
        se = [df[df.arm == a][met].std() / max(1, np.sqrt((df.arm == a).sum())) for a in arms]
        ax.bar(range(len(arms)), mu, yerr=se, capsize=4,
               color=[PALETTE[i] for i in range(len(arms))], edgecolor="black", lw=0.8)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(["none", "spec", "admit", "both"], rotation=0)
        ax.set_title(met.replace("gamma_", r"$\gamma$ ").replace("_", " "))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_rq2_boundary(df: pd.DataFrame, path: str):
    """Stability boundary as line plots (oscillation fraction vs gain)."""
    set_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, pr in enumerate(sorted(df.period_ratio.unique())):
        d = df[df.period_ratio == pr].groupby("gain")["oscillatory"].mean()
        ax.plot(d.index, d.values, color=PALETTE[i % len(PALETTE)],
                marker=MARKERS[i % len(MARKERS)], label=rf"$T_a/T_s={pr:g}$")
    ax.set_xscale("log")
    ax.set_xlabel("controller gain")
    ax.set_ylabel("fraction of runs oscillatory")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(path)
    plt.close(fig)


def fig_rq3_cost(df: pd.DataFrame, path: str):
    set_style()
    order = ["no_spec", "static_spec_static_admit", "adaptive_spec_only",
             "adaptive_admit_only", "both_naive"]
    short = ["no spec", "static", "spec only", "admit only", "both"]
    mets = [("slo_attainment", "SLO attainment"), ("tpot_p99", "P99 TPOT (s)"),
            ("goodput_tok_s", "goodput (tok/s)"), ("gpu_s_per_ktok", "GPU-s / ktok")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    for ax, (met, lab) in zip(axes, mets):
        mu = [df[df.arm == a][met].mean() for a in order]
        se = [df[df.arm == a][met].std() / max(1, np.sqrt((df.arm == a).sum())) for a in order]
        ax.bar(range(len(order)), mu, yerr=se, capsize=4,
               color=[PALETTE[i] for i in range(len(order))], edgecolor="black", lw=0.8)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(short, rotation=30, ha="right")
        ax.set_ylabel(lab)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_rq4_coordination(df: pd.DataFrame, path: str):
    set_style()
    order = ["naive", "timescale", "hysteresis", "mimo"]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4))
    mets = [("gamma_spectral_concentration", r"$\gamma$ spectral conc."),
            ("slo_attainment", "SLO attainment"),
            ("goodput_tok_s", "goodput (tok/s)")]
    for ax, (met, lab) in zip(axes, mets):
        mu = [df[df.coord_name == c][met].mean() for c in order]
        se = [df[df.coord_name == c][met].std() / max(1, np.sqrt((df.coord_name == c).sum()))
              for c in order]
        ax.bar(range(len(order)), mu, yerr=se, capsize=4,
               color=[PALETTE[i] for i in range(len(order))], edgecolor="black", lw=0.8)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=25, ha="right")
        ax.set_ylabel(lab)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ==========================================================================
# LaTeX tables (booktabs + cellcolor)
# ==========================================================================


def _cellcolor(v: float, vmin: float, vmax: float, invert: bool = False,
               color: str = "red") -> str:
    if not np.isfinite(v) or vmax <= vmin:
        return ""
    f = (v - vmin) / (vmax - vmin)
    if invert:
        f = 1.0 - f
    return rf"\cellcolor{{{color}!{int(np.clip(f, 0, 1) * 45):d}}}"


def table_stability_map(df: pd.DataFrame, path: str,
                        caption: str = "Fraction of runs classified oscillatory.",
                        label: str = "tab:stability-map"):
    """RQ2 stability map as a booktabs table with cellcolor shading."""
    gains = sorted(df.gain.unique())
    ratios = sorted(df.period_ratio.unique())
    piv = df.pivot_table(index="period_ratio", columns="gain",
                         values="oscillatory", aggfunc="mean")
    lines = [r"\begin{table}[t]", r"\centering",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\small",
             r"\begin{tabular}{l" + "r" * len(gains) + "}", r"\toprule",
             r"$T_a/T_s$ & " + " & ".join(rf"$g{{=}}{g:g}$" for g in gains) + r" \\",
             r"\midrule"]
    for r_ in ratios:
        cells = []
        for g in gains:
            v = piv.loc[r_, g] if (r_ in piv.index and g in piv.columns) else np.nan
            cells.append(_cellcolor(v, 0.0, 1.0) + (f"{v:.2f}" if np.isfinite(v) else "--"))
        lines.append(f"${r_:g}$ & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write(path, "\n".join(lines))


def table_e2e(df: pd.DataFrame, path: str, group: str, order: Sequence[str],
              caption: str, label: str,
              metrics=(("slo_attainment", "SLO attain.", True),
                       ("ttft_p99", "P99 TTFT (s)", False),
                       ("tpot_p99", "P99 TPOT (s)", False),
                       ("goodput_tok_s", "Goodput (tok/s)", True),
                       ("gpu_s_per_ktok", "GPU-s/ktok", False),
                       ("gamma_spectral_concentration", r"$\gamma$ spec.\ conc.", False))):
    """E2E table, best value per column highlighted green."""
    lines = [r"\begin{table*}[t]", r"\centering",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\small",
             r"\begin{tabular}{l" + "r" * len(metrics) + "}", r"\toprule",
             "Configuration & " + " & ".join(m[1] for m in metrics) + r" \\",
             r"\midrule"]
    stats = {g: df[df[group] == g] for g in order}
    for met, _, higher_better in metrics:
        pass
    best = {}
    for met, _, hb in metrics:
        vals = {g: stats[g][met].mean() for g in order if len(stats[g])}
        if vals:
            best[met] = max(vals, key=vals.get) if hb else min(vals, key=vals.get)
    for g in order:
        d = stats.get(g)
        if d is None or d.empty:
            continue
        cells = []
        for met, _, hb in metrics:
            mu, sd = d[met].mean(), d[met].std()
            hl = r"\cellcolor{green!18}" if best.get(met) == g else ""
            cells.append(rf"{hl}{mu:.3g}\,$\pm$\,{sd:.2g}")
        lines.append(g.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    _write(path, "\n".join(lines))


def table_controller_ablation(df: pd.DataFrame, path: str,
                              caption: str = "Interference signature per speculation controller.",
                              label: str = "tab:controller-ablation"):
    specs = sorted(df.spec_variant.unique())
    admits = sorted(df.admit_variant.unique())
    lines = [r"\begin{table}[t]", r"\centering",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\small",
             r"\begin{tabular}{ll" + "rr" + "}", r"\toprule",
             r"$L_{\mathrm{spec}}$ & $L_{\mathrm{admit}}$ & isolated & composed \\",
             r"\midrule"]
    met = "gamma_spectral_concentration"
    for s in specs:
        for a in admits:
            iso = df[(df.spec_variant == s) & (df.admit_variant == a) &
                     (df.condition == "isolated")][met].mean()
            com = df[(df.spec_variant == s) & (df.admit_variant == a) &
                     (df.condition == "composed")][met].mean()
            hl = r"\cellcolor{red!20}" if np.isfinite(com) and np.isfinite(iso) and com > 2 * max(iso, 1e-6) else ""
            lines.append(rf"{s} & {a} & {iso:.3f} & {hl}{com:.3f} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write(path, "\n".join(lines))


def _write(path: str, content: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(content + "\n")
