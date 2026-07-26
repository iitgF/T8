"""Ablation: how much of the study's result rests on Lending Club's own pricing?

The main study keeps grade, sub_grade and int_rate, which is the information
set an investor browsing the platform actually sees. Those three are largely
one variable in three notations, and they compress the differences between
models: every learner spends its capacity re-ranking a population Lending
Club has already ranked.

This script reruns the identical protocol without them, so every model must
build its risk ordering from the credit bureau alone. Two variants are run
because installment is a deterministic function of loan amount, term and
rate; with the term fixed at 36 months it can reintroduce the rate.

    variant A   drop grade, sub_grade, int_rate
    variant B   also drop installment

Results are written to results/*_nopricing_{A,B}.csv and left beside the
main study's files, which this script never touches.

Usage:  python experiment_no_pricing.py
"""

import json
import time

import numpy as np
import pandas as pd

from creditrisk.calibration import calibrate
from creditrisk.config import CFG
from creditrisk.data import (CATEGORICAL, FEATURES, NUMERIC,
                             load_lending_club, split_by_vintage)
from creditrisk.decision import best_operating_point, loan_pnl, profit_curve
from creditrisk.evaluate import leaderboard
from creditrisk.metrics import auc, bootstrap_ci, delong_test
from creditrisk.pipeline import RESULTS, _fit_all

PRICING = ["int_rate", "grade", "sub_grade"]
VARIANTS = {"A": PRICING, "B": PRICING + ["installment"]}


def run_variant(tag, drop, train, val, test, cfg):
    numeric = [c for c in NUMERIC if c not in drop]
    categorical = [c for c in CATEGORICAL if c not in drop]
    feats = numeric + categorical
    y_tr, y_val, y_te = (train["target"].to_numpy(), val["target"].to_numpy(),
                         test["target"].to_numpy())

    print(f"\n--- variant {tag}: dropped {drop} -> {len(feats)} features ---")
    preds, fitted, _ = _fit_all(cfg, numeric, categorical, train[feats], y_tr,
                                val[feats], y_val, test[feats])

    per_run, agg = leaderboard(preds, y_val, y_te, cfg)
    agg.to_csv(RESULTS / f"leaderboard_lc_nopricing_{tag}.csv", index=False)

    # significance of each challenger against the scorecard
    p_sc = preds["scorecard"][cfg.seed][1]
    rows = []
    for name in ["xgboost", "lightgbm", "mlp"]:
        d, p = delong_test(y_te, preds[name][cfg.seed][1], p_sc)
        lo, hi = bootstrap_ci(y_te, preds[name][cfg.seed][1], p_sc, auc,
                              n_boot=cfg.n_bootstrap, seed=cfg.seed)
        rows.append(dict(model=name, auc_diff_vs_scorecard=d, delong_p=p,
                         ci_lo=lo, ci_hi=hi))
    sig = pd.DataFrame(rows)
    sig.to_csv(RESULTS / f"significance_lc_nopricing_{tag}.csv", index=False)

    # profit at the best cutoff, isotonic-calibrated, same cash-flow model
    pnl = loan_pnl(y_te, test["loan_amnt"], test["installment"], cfg.lgd)
    ops = []
    for name in preds:
        p_iso = calibrate("isotonic", preds[name][cfg.seed][0], y_val,
                          preds[name][cfg.seed][1])
        best = best_operating_point(profit_curve(p_iso, pnl, cfg))
        ops.append(dict(model=name, cutoff=best["cutoff"],
                        accept_rate=best["accept_rate"],
                        profit_per_1000=best["profit_per_1000"]))
    ops = pd.DataFrame(ops)
    ops.to_csv(RESULTS / f"operating_points_lc_nopricing_{tag}.csv", index=False)
    return agg, sig, ops


def main():
    t0 = time.time()
    cfg = CFG
    df = load_lending_club(cfg)
    train, val, test = split_by_vintage(df, cfg)

    # Is installment a proxy for the interest rate? With the term fixed at 36
    # months, the monthly payment per dollar borrowed is a monotone function
    # of the rate, so a high correlation here means variant A leaks pricing.
    ratio = test["installment"] / test["loan_amnt"]
    r = np.corrcoef(ratio, test["int_rate"])[0, 1]
    print(f"corr(installment / loan_amnt, int_rate) = {r:.4f}")

    out = {}
    for tag, drop in VARIANTS.items():
        out[tag] = run_variant(tag, drop, train, val, test, cfg)

    print("\n================ SUMMARY ================")
    base = pd.read_csv(RESULTS / "leaderboard_lc.csv")
    base = base[base.calibration == "raw"].set_index("model")["auc_mean"]
    for tag, (agg, sig, ops) in out.items():
        a = agg[agg.calibration == "raw"].set_index("model")["auc_mean"]
        cmp = pd.DataFrame({"baseline_auc": base, f"no_pricing_{tag}": a})
        cmp["change"] = cmp[f"no_pricing_{tag}"] - cmp["baseline_auc"]
        print(f"\nvariant {tag}:")
        print(cmp.round(4).to_string())
        print("  vs scorecard:")
        print(sig.round(5).to_string(index=False))
    print(f"\ntotal runtime {time.time() - t0:.0f}s")
    (RESULTS / "summary_nopricing.json").write_text(json.dumps(
        {"corr_installment_ratio_vs_int_rate": float(r),
         "variants": {k: v[0].to_dict(orient="records") for k, v in out.items()}},
        indent=2))


if __name__ == "__main__":
    main()
