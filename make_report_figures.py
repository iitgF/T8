"""Render the two report figures at single-column width.

The figures in figures/ are sized for slides (6.6 in wide). Placing them in
an 8.3 cm report column would scale their text down to roughly 3.5 pt. This
script re-renders them at 3.3 in with type sized for that width, writing
fig_reliability_col.png and fig_profit_col.png.

The profit curves are read from results/. The reliability curves are not
saved by the pipeline, so the first run refits the seed-42 models to obtain
them and caches the result in results/reliability_lc.csv; later runs reuse
the cache and finish instantly.

Usage:  python make_report_figures.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from creditrisk.calibration import calibrate
from creditrisk.config import CFG
from creditrisk.figures import GOOD, GRID, INK, MODEL_COLORS, MODEL_LABELS, MUTED, AXIS
from creditrisk.metrics import reliability_curve
from creditrisk.pipeline import FIGURES, RESULTS

CACHE = RESULTS / "reliability_lc.csv"


def column_style():
    """Type sized so that a 3.3 in figure is legible at 1:1 in the report."""
    plt.rcParams.update({
        "figure.dpi": 400, "savefig.dpi": 400, "savefig.bbox": "tight",
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 1.0,
    })


def build_reliability_cache():
    from creditrisk.data import (CATEGORICAL, FEATURES, NUMERIC,
                                 load_lending_club, split_by_vintage)
    from creditrisk.pipeline import _fit_all

    print("no cache: refitting seed-42 models to recover reliability curves ...")
    df = load_lending_club(CFG)
    train, val, test = split_by_vintage(df, CFG)
    y_tr, y_val, y_te = (train["target"].to_numpy(), val["target"].to_numpy(),
                         test["target"].to_numpy())
    cfg = CFG.__class__(**{**CFG.__dict__, "n_seeds": 1})
    preds, _, _ = _fit_all(cfg, NUMERIC, CATEGORICAL, train[FEATURES], y_tr,
                           val[FEATURES], y_val, test[FEATURES])
    rows = []
    for name in preds:
        p_val, p_raw = preds[name][cfg.seed]
        p_iso = calibrate("isotonic", p_val, y_val, p_raw)
        for kind, p in [("raw", p_raw), ("isotonic", p_iso)]:
            c = reliability_curve(y_te, p, CFG.ece_bins)
            c.insert(0, "calibration", kind)
            c.insert(0, "model", name)
            rows.append(c)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(CACHE, index=False)
    return out


def fig_reliability(rel, path):
    column_style()
    fig, axes = plt.subplots(2, 2, figsize=(3.3, 2.9), sharex=True, sharey=True)
    for ax, model in zip(axes.ravel(), MODEL_COLORS):
        lim = 0.0
        for kind, colour, marker in [("raw", MUTED, "o"),
                                     ("isotonic", MODEL_COLORS[model], "s")]:
            d = rel[(rel.model == model) & (rel.calibration == kind)]
            ax.plot(d["p_mean"], d["y_rate"], marker=marker, ms=2.2,
                    color=colour, label=kind.capitalize())
            lim = max(lim, d["p_mean"].max(), d["y_rate"].max())
        lim *= 1.08
        ax.plot([0, lim], [0, lim], ls="--", lw=0.6, color=AXIS, zorder=0)
        ax.set_title(MODEL_LABELS[model], color=INK, pad=2)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xticks([0, 0.1, 0.2, 0.3])
        ax.set_yticks([0, 0.1, 0.2, 0.3])
    for ax in axes[1]:
        ax.set_xlabel("Mean predicted PD")
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed rate")
    axes[0, 0].legend(loc="upper left", handlelength=1.2, borderpad=0.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path.name)


def fig_profit(curves, accept_all, path):
    column_style()
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(3.3, 1.55),
                                  width_ratios=[1.1, 1.0])
    peak = curves["profit_per_1000"].max()
    for panel, xlim, ylim in [((ax), (0.0, 0.6), None),
                              ((axz), (0.15, 0.45),
                               (accept_all * 0.992, peak * 1.004))]:
        for model, g in curves.groupby("model"):
            g = g.sort_values("cutoff")
            panel.plot(g["cutoff"], g["profit_per_1000"] / 1000,
                       color=MODEL_COLORS[model], label=MODEL_LABELS[model])
            best = g.loc[g["profit_per_1000"].idxmax()]
            panel.plot(best["cutoff"], best["profit_per_1000"] / 1000, "o",
                       ms=2.6, color=MODEL_COLORS[model], mec="white", mew=0.5)
        panel.axhline(accept_all / 1000, ls="--", lw=0.6, color=MUTED)
        panel.set_xlim(*xlim)
        if ylim:
            panel.set_ylim(ylim[0] / 1000, ylim[1] / 1000)
        panel.set_xlabel("PD cutoff")
    ax.set_ylabel("Profit / 1,000 (\\$k)")
    ax.set_title("Full sweep", color=INK, pad=2)
    axz.set_title("Peak region", color=INK, pad=2)
    ax.legend(loc="lower right", handlelength=1.0, borderpad=0.2, labelspacing=0.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path.name)


def main():
    rel = pd.read_csv(CACHE) if CACHE.exists() else build_reliability_cache()
    fig_reliability(rel, FIGURES / "fig_reliability_col.png")

    curves = pd.read_csv(RESULTS / "profit_curves_lc.csv")
    import json
    accept_all = json.loads((RESULTS / "summary_lc.json").read_text())[
        "accept_all_profit_per_1000"]
    fig_profit(curves, accept_all, FIGURES / "fig_profit_col.png")


if __name__ == "__main__":
    main()
