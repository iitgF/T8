"""Teaching figures: what AUC is, and what isotonic calibration does.

Neither figure appears in the four-page report, which has no room. They exist
to explain two ideas the report relies on, using this study's own numbers
rather than a textbook illustration, and are useful for the video and for
answering questions about the method.

  fig_roc_explained.png        the ROC curve with the area shaded, plus both
                               models on one pair of axes
  fig_isotonic_explained.png   the calibration map fitted on the 2014 vintage,
                               and its effect on the 2015 reliability curve

Refits the scorecard and XGBoost on the reference seed, about three minutes.

Usage:  python make_explainer_figures.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from creditrisk.calibration import IsotonicCalibrator
from creditrisk.config import CFG
from creditrisk.data import (CATEGORICAL, FEATURES, NUMERIC, load_lending_club,
                             split_by_vintage)
from creditrisk.figures import AXIS, GRID, MUTED
from creditrisk.metrics import auc, ece, reliability_curve
from creditrisk.models import DesignMatrix, fit_xgboost
from creditrisk.pipeline import FIGURES
from creditrisk.scorecard import Scorecard

AQUA = "#1baf7a"
DEEP = "#0d6b48"
NAVY = "#1B3A6B"   # matches the deck and the report figures


def style():
    plt.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.7,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    })


def fig_roc(y, p_xgb, p_sc, path):
    style()
    fx, tx, _ = roc_curve(y, p_xgb)
    fs, ts, _ = roc_curve(y, p_sc)
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 4.0))

    a.fill_between(fx, 0, tx, color=AQUA, alpha=0.28, lw=0,
                   label=f"area = AUC = {auc(y, p_xgb):.4f}")
    a.plot(fx, tx, color=AQUA, lw=1.8, label="XGBoost ROC curve")
    a.plot([0, 1], [0, 1], ls="--", lw=1.0, color=MUTED)
    a.annotate("random model:\ncurve on the diagonal,\narea = 0.5",
               xy=(0.78, 0.78), xytext=(0.88, 0.44), fontsize=8, color=MUTED,
               ha="center", arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    a.text(0.34, 0.17, "shaded region\nis the AUC", fontsize=10, color=DEEP,
           ha="center", fontweight="bold")
    a.set_title("Where the area is", color=NAVY, fontweight="bold")

    b.plot(fx, tx, color=AQUA, lw=1.8, label=f"XGBoost  {auc(y, p_xgb):.4f}")
    b.plot(fs, ts, color=NAVY, lw=1.8, label=f"Scorecard  {auc(y, p_sc):.4f}")
    b.plot([0, 1], [0, 1], ls="--", lw=1.0, color=MUTED, label="Random  0.5000")
    b.set_title("The two models, same axes", color=NAVY, fontweight="bold")

    for ax in (a, b):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("False positive rate\n(good borrowers wrongly flagged)")
        ax.set_ylabel("True positive rate\n(defaulters correctly flagged)")
        ax.set_aspect("equal")
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path.name)


def fig_isotonic(y_te, p_te, cal, path):
    style()
    p_iso = cal.transform(p_te)
    lo, hi = float(np.percentile(p_te, 0.5)), float(np.percentile(p_te, 99.5))
    grid = np.linspace(lo, hi, 800)
    mapped = cal.transform(grid)

    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.9))
    a.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color=MUTED,
           label="identity (no change)")
    a.plot(grid, mapped, color=AQUA, lw=2.0, label="isotonic map, fitted on 2014")
    a.fill_between(grid, grid, mapped, where=mapped >= grid, color=AQUA,
                   alpha=0.18, lw=0, label="correction applied")
    a.set_xlabel("Raw predicted PD from the model")
    a.set_ylabel("Calibrated PD after the map")
    a.set_title("The map it learns", color=NAVY, fontweight="bold")
    mid = lo + 0.55 * (hi - lo)
    a.annotate("the curve sits above the identity:\nevery prediction is pushed up,\n"
               "because the model under-predicts",
               xy=(mid, float(cal.transform(np.array([mid]))[0])),
               xytext=(lo + 0.30 * (hi - lo), hi * 0.34), fontsize=8,
               color=NAVY, ha="center",
               arrowprops=dict(arrowstyle="->", color=NAVY, lw=0.8))
    a.legend(loc="upper left")

    for kind, p, colour, marker in [("raw", p_te, MUTED, "o"),
                                    ("isotonic", p_iso, AQUA, "s")]:
        r = reliability_curve(y_te, p, CFG.ece_bins)
        b.plot(r["p_mean"], r["y_rate"], marker=marker, ms=4, color=colour,
               lw=1.4, label=f"{kind}   ECE {ece(y_te, p, CFG.ece_bins):.4f}")
    lim = 0.33
    b.plot([0, lim], [0, lim], ls="--", lw=1.0, color=MUTED)
    b.set_xlim(0, lim)
    b.set_ylim(0, lim)
    b.set_xlabel("Mean predicted PD")
    b.set_ylabel("Observed default rate")
    b.set_title("What it fixes, XGBoost on 2015", color=NAVY, fontweight="bold")
    b.set_aspect("equal")
    b.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path.name)
    print(f"  AUC {auc(y_te, p_te):.4f} -> {auc(y_te, p_iso):.4f} "
          f"(monotone maps preserve ranking)")
    print(f"  ECE {ece(y_te, p_te, CFG.ece_bins):.4f} -> "
          f"{ece(y_te, p_iso, CFG.ece_bins):.4f}")
    print(f"  {len(np.unique(mapped))} distinct outputs from {len(grid)} inputs")


def main():
    df = load_lending_club(CFG)
    tr, va, te = split_by_vintage(df, CFG)
    y_tr, y_va, y_te = (tr.target.to_numpy(), va.target.to_numpy(),
                        te.target.to_numpy())

    sc = Scorecard(CFG, NUMERIC, CATEGORICAL).fit(tr[FEATURES], y_tr)
    p_sc = sc.predict_proba(te[FEATURES])

    dm = DesignMatrix(NUMERIC, CATEGORICAL).fit(tr[FEATURES])
    xgb = fit_xgboost(CFG, CFG.seed, dm.transform(tr[FEATURES]), y_tr,
                      dm.transform(va[FEATURES]), y_va)
    p_va = xgb.predict_proba(dm.transform(va[FEATURES]))[:, 1]
    p_te = xgb.predict_proba(dm.transform(te[FEATURES]))[:, 1]

    fig_roc(y_te, p_te, p_sc, FIGURES / "fig_roc_explained.png")
    fig_isotonic(y_te, p_te, IsotonicCalibrator().fit(p_va, y_va),
                 FIGURES / "fig_isotonic_explained.png")


if __name__ == "__main__":
    main()
