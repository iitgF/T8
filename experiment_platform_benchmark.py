"""Benchmark the study's models against Lending Club's own production model.

The main study builds a weight-of-evidence scorecard from scratch, because a
real bank scorecard is proprietary and, even if obtainable, would have been
fitted on a different population. But the data does contain the output of one
real production underwriting model: Lending Club's own sub-grade, assigned
before the loan was listed.

This script treats that grade as a competing model and scores it on the same
2015 test vintage under the same criteria as everything else:

  ranking      the ordinal sub-grade used directly as a risk score
  as a PD      each sub-grade mapped to its observed training default rate,
               then recalibrated on the 2014 vintage exactly like every other
               model, which yields probabilities and therefore Brier, ECE and
               realized profit

Writes results/platform_benchmark.csv and results/platform_significance.csv.

Usage:  python experiment_platform_benchmark.py
"""

import numpy as np
import pandas as pd

from creditrisk.calibration import calibrate
from creditrisk.config import CFG
from creditrisk.data import (CATEGORICAL, FEATURES, NUMERIC, load_lending_club,
                             split_by_vintage)
from creditrisk.decision import best_operating_point, loan_pnl, profit_curve
from creditrisk.metrics import auc, brier, delong_test, ece, ks_stat
from creditrisk.pipeline import RESULTS, _fit_all

SMOOTH = 50.0   # prior strength when mapping a grade to its default rate


def ordinal(series, levels):
    return series.map({g: i for i, g in enumerate(levels)}).astype(float)


def grade_to_pd(train, col, y_tr):
    """Observed default rate per level on train, smoothed toward the mean."""
    base = y_tr.mean()
    g = pd.DataFrame({col: train[col], "y": y_tr}).groupby(col)["y"]
    rate = (g.sum() + SMOOTH * base) / (g.count() + SMOOTH)
    return rate.to_dict(), base


def main():
    df = load_lending_club(CFG)
    train, val, test = split_by_vintage(df, CFG)
    y_tr, y_val, y_te = (train["target"].to_numpy(), val["target"].to_numpy(),
                         test["target"].to_numpy())
    pnl = loan_pnl(y_te, test["loan_amnt"], test["installment"], CFG.lgd)

    rows, preds = [], {}

    # --- Lending Club's own model, three encodings of the same judgement ----
    for col in ["sub_grade", "grade"]:
        levels = sorted(df[col].dropna().unique())
        s_te = ordinal(test[col], levels).fillna(-1)
        mapping, base = grade_to_pd(train, col, y_tr)
        p_val = val[col].map(mapping).fillna(base).to_numpy()
        p_te = test[col].map(mapping).fillna(base).to_numpy()
        p_iso = calibrate("isotonic", p_val, y_val, p_te)
        preds[f"platform_{col}"] = p_te
        best = best_operating_point(profit_curve(p_iso, pnl, CFG))
        rows.append(dict(model=f"Lending Club {col}", auc=auc(y_te, s_te),
                         ks=ks_stat(y_te, s_te), brier_iso=brier(y_te, p_iso),
                         ece_iso=ece(y_te, p_iso, CFG.ece_bins),
                         profit=best["profit_per_1000"]))

    s_te = test["int_rate"].fillna(test["int_rate"].median()).to_numpy()
    preds["platform_int_rate"] = s_te
    rows.append(dict(model="Lending Club int_rate", auc=auc(y_te, s_te),
                     ks=ks_stat(y_te, s_te), brier_iso=np.nan, ece_iso=np.nan,
                     profit=np.nan))

    # --- this project's models, seed 42, for a like-for-like comparison -----
    cfg = CFG.__class__(**{**CFG.__dict__, "n_seeds": 1})
    fitted, _, _ = _fit_all(cfg, NUMERIC, CATEGORICAL, train[FEATURES], y_tr,
                            val[FEATURES], y_val, test[FEATURES])
    for name in ["scorecard", "xgboost", "lightgbm", "mlp"]:
        p_val, p_te = fitted[name][cfg.seed]
        p_iso = calibrate("isotonic", p_val, y_val, p_te)
        preds[name] = p_te
        best = best_operating_point(profit_curve(p_iso, pnl, CFG))
        rows.append(dict(model=name, auc=auc(y_te, p_te), ks=ks_stat(y_te, p_te),
                         brier_iso=brier(y_te, p_iso),
                         ece_iso=ece(y_te, p_iso, CFG.ece_bins),
                         profit=best["profit_per_1000"]))

    tbl = pd.DataFrame(rows).sort_values("auc", ascending=False)
    tbl.to_csv(RESULTS / "platform_benchmark.csv", index=False)
    print(tbl.round(4).to_string(index=False))

    # --- significance against the platform's own model ----------------------
    ref = preds["platform_sub_grade"]
    sig = []
    for name in ["xgboost", "lightgbm", "scorecard", "mlp"]:
        d, p = delong_test(y_te, preds[name], ref)
        sig.append(dict(model=name, auc_diff_vs_platform=d, delong_p=p))
    sig = pd.DataFrame(sig)
    sig.to_csv(RESULTS / "platform_significance.csv", index=False)
    print("\nversus Lending Club's own sub-grade:")
    print(sig.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
