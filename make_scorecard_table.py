"""Export the fitted scorecard as a points table, the artifact a lender uses.

The pipeline fits the scorecard, scores with it and discards it. This script
refits it on the training vintages and writes the model itself: one row per
feature and bin, with the points that bin contributes. An applicant's score
is the sum of their rows, exactly, with no approximation and no post-hoc
explainer.

The allocation follows the standard construction. The logistic models the
log-odds of default,

    ln(odds_bad) = b0 + sum_i beta_i * WoE_i,

and the score is an affine function of the log-odds of repayment:

    score = offset + factor * ln(odds_good) = offset - factor * ln(odds_bad),

with factor = PDO / ln 2 and offset = base_score - factor * ln(base_odds).
Substituting gives an additive decomposition over features,

    points_ij = -factor * beta_i * WoE_ij + (offset - factor * b0) / n,

where the second term spreads the intercept evenly across the n features so
the per-feature points sum to the applicant's total score.

Writes results/scorecard_points.csv and prints a worked example.

Usage:  python make_scorecard_table.py
"""

import numpy as np
import pandas as pd

from creditrisk.config import CFG
from creditrisk.data import (CATEGORICAL, FEATURES, NUMERIC, load_lending_club,
                             split_by_vintage)
from creditrisk.pipeline import RESULTS
from creditrisk.scorecard import Scorecard
from creditrisk.woe import MISSING, OTHER


def bin_label(feature, key, binner):
    """Readable bin description: intervals for numerics, levels otherwise."""
    if key == MISSING:
        return "missing"
    if key == OTHER:
        return "other (rare levels pooled)"
    if feature not in binner.numeric:
        return str(key)
    # pandas Interval strings such as "(-inf, 11.32]" are already readable
    return str(key).replace("-inf", "min").replace("inf", "max")


def build_table(sc: Scorecard, cfg) -> pd.DataFrame:
    factor = cfg.pdo / np.log(2)
    offset = cfg.base_score - factor * np.log(cfg.base_odds)
    feats = sc.features_
    beta = dict(zip(feats, sc.lr.coef_[0]))
    b0 = float(sc.lr.intercept_[0])
    base_share = (offset - factor * b0) / len(feats)

    rows = []
    for f in feats:
        for key, woe in sc.binner.woe_maps_[f].items():
            rows.append(dict(
                feature=f,
                bin=bin_label(f, key, sc.binner),
                woe=round(float(woe), 4),
                points=round(-factor * beta[f] * float(woe) + base_share, 1),
            ))
    tbl = pd.DataFrame(rows)
    # order features by information value, bins by points within a feature
    iv = sc.iv_table().set_index("feature")["iv"]
    tbl["iv"] = tbl["feature"].map(iv).round(4)
    tbl = tbl.sort_values(["iv", "feature", "points"],
                          ascending=[False, True, False])
    return tbl[["feature", "iv", "bin", "woe", "points"]].reset_index(drop=True)


def worked_example(sc, tbl, X, cfg):
    """Show that summing one applicant's rows reproduces .score() exactly."""
    row = X.iloc[[0]]
    total = 0.0
    lines = []
    for f in sc.features_:
        if f in sc.binner.numeric:
            labels = sc.binner._bin_labels_numeric(row[f], sc.binner.edges_[f])
            key = labels.iloc[0]
        else:
            v = str(row[f].fillna(MISSING).iloc[0])
            key = v if v in sc.binner.woe_maps_[f] else OTHER
        pts = tbl[(tbl.feature == f) & (tbl.bin == bin_label(f, key, sc.binner))]
        p = float(pts["points"].iloc[0])
        total += p
        lines.append(f"  {f:<16} {bin_label(f, key, sc.binner):<28} {p:>7.1f}")
    print("\nWorked example, first test applicant:")
    print("\n".join(lines))
    print(f"  {'':<16} {'TOTAL':<28} {total:>7.1f}")
    print(f"  {'':<16} {'Scorecard.score()':<28} "
          f"{float(sc.score(row)[0]):>7.1f}")


def main():
    df = load_lending_club(CFG)
    train, val, test = split_by_vintage(df, CFG)
    sc = Scorecard(CFG, NUMERIC, CATEGORICAL).fit(train[FEATURES],
                                                  train["target"].to_numpy())
    tbl = build_table(sc, CFG)
    out = RESULTS / "scorecard_points.csv"
    tbl.to_csv(out, index=False)
    print(f"{len(tbl)} rows across {tbl.feature.nunique()} features -> {out.name}")

    print("\nPoints range per feature (wider = more influence on the score):")
    span = (tbl.groupby("feature")["points"].agg(["min", "max"])
            .assign(span=lambda d: (d["max"] - d["min"]).round(1))
            .sort_values("span", ascending=False).round(1))
    print(span.to_string())

    worked_example(sc, tbl, test[FEATURES], CFG)


if __name__ == "__main__":
    main()
