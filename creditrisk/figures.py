"""Report figures (matplotlib, print-oriented, light surface).

Colors follow the entity: each model keeps one fixed hue everywhere
(categorical slots in fixed order); good/bad outcome uses the blue/red
diverging pair; chrome (grid, axes, labels) stays recessive.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# fixed categorical assignment, one hue per model across every figure
MODEL_COLORS = {
    "scorecard": "#2a78d6",   # blue
    "xgboost": "#1baf7a",     # aqua
    "lightgbm": "#eda100",    # yellow
    "mlp": "#008300",         # green
}
MODEL_LABELS = {
    "scorecard": "Scorecard",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "mlp": "MLP",
}
GOOD, BAD = "#2a78d6", "#d03b3b"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 1.4,
    })


def fig_reliability(rel: dict, path):
    """2x2 small multiples: raw vs isotonic reliability per model."""
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.2), sharex=True, sharey=True)
    for ax, model in zip(axes.ravel(), MODEL_COLORS):
        lim = 0.0
        for name, color, marker in [("raw", MUTED, "o"),
                                    ("isotonic", MODEL_COLORS[model], "s")]:
            df = rel[model][name]
            ax.plot(df["p_mean"], df["y_rate"], marker=marker, ms=3.5,
                    color=color, label=name.capitalize())
            lim = max(lim, df["p_mean"].max(), df["y_rate"].max())
        lim *= 1.1
        ax.plot([0, lim], [0, lim], ls="--", lw=0.8, color=AXIS, zorder=0)
        ax.set_title(MODEL_LABELS[model], color=INK)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
    for ax in axes[1]:
        ax.set_xlabel("Mean predicted PD")
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed default rate")
    axes[0, 0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_profit(curves: pd.DataFrame, accept_all: float, path):
    """Profit per 1,000 applicants vs PD cutoff: full sweep + peak zoom."""
    setup_style()
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(6.6, 3.0),
                                  width_ratios=[1.15, 1.0])
    peak = curves["profit_per_1000"].max()
    for panel, xlim, ylim in [
        (ax, (0.0, 0.6), None),
        (axz, (0.15, 0.45), (accept_all * 0.99, peak * 1.004)),
    ]:
        for model, g in curves.groupby("model"):
            g = g.sort_values("cutoff")
            panel.plot(g["cutoff"], g["profit_per_1000"],
                       color=MODEL_COLORS[model], label=MODEL_LABELS[model])
            best = g.loc[g["profit_per_1000"].idxmax()]
            panel.plot(best["cutoff"], best["profit_per_1000"], "o", ms=4.5,
                       color=MODEL_COLORS[model], mec="white", mew=0.8)
        panel.axhline(accept_all, ls="--", lw=0.9, color=MUTED)
        panel.set_xlim(*xlim)
        if ylim:
            panel.set_ylim(*ylim)
        panel.set_xlabel("PD acceptance cutoff")
    ax.annotate("accept everyone", xy=(0.59, accept_all),
                xytext=(0, 4), textcoords="offset points",
                ha="right", fontsize=7, color=MUTED)
    ax.set_ylabel("Profit per 1,000 applicants (USD)")
    ax.set_title("Full sweep", color=INK)
    axz.set_title("Peak region (zoom)", color=INK)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_score_dist(scores: np.ndarray, y: np.ndarray, path):
    """Scorecard score distributions for repaid vs defaulted loans."""
    setup_style()
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    bins = np.linspace(scores.min(), scores.max(), 60)
    ax.hist(scores[y == 0], bins=bins, density=True, alpha=0.75,
            color=GOOD, label="Fully paid", edgecolor="white", lw=0.2)
    ax.hist(scores[y == 1], bins=bins, density=True, alpha=0.75,
            color=BAD, label="Charged off", edgecolor="white", lw=0.2)
    ax.set_xlabel("Scorecard score (higher = safer)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_importance(imp: pd.DataFrame, path):
    """Side-by-side: XGBoost importance vs scorecard information value."""
    setup_style()
    imp = imp.sort_values("xgb_importance")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.6), sharey=True)
    axes[0].barh(imp["feature"], imp["xgb_importance"],
                 color=MODEL_COLORS["xgboost"], height=0.65)
    axes[0].set_title("XGBoost importance (share)", color=INK)
    axes[1].barh(imp["feature"], imp["iv"],
                 color=MODEL_COLORS["scorecard"], height=0.65)
    axes[1].set_title("Scorecard information value", color=INK)
    for ax in axes:
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
