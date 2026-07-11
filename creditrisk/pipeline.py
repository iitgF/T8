"""End-to-end orchestration: data -> models -> calibration -> evaluation.

`run_lending_club` is the main study; `run_taiwan` repeats the discrimination
and calibration comparison on an independent population as a robustness check
(no profit analysis there: the UCI file has no loan-level cash flows).
"""

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import figures
from .calibration import calibrate
from .config import CFG, Config, ROOT
from .data import (CATEGORICAL, FEATURES, NUMERIC, TAIWAN_CATEGORICAL,
                   load_lending_club, load_taiwan, split_by_vintage)
from .decision import best_operating_point, loan_pnl, profit_curve
from .evaluate import leaderboard
from .metrics import auc, bootstrap_ci, brier, delong_test, ece, reliability_curve
from .models import DesignMatrix, fit_lightgbm, fit_mlp, fit_xgboost
from .scorecard import Scorecard

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

FITTERS = {"xgboost": fit_xgboost, "lightgbm": fit_lightgbm, "mlp": fit_mlp}


def _proba(model, X) -> np.ndarray:
    p = model.predict_proba(X)
    return p[:, 1] if p.ndim == 2 else p


def _fit_all(cfg, numeric, categorical, X_tr, y_tr, X_val, y_val, X_te):
    """Fit scorecard + ML challengers; return predictions and first-seed models."""
    preds, fitted = {}, {}

    sc = Scorecard(cfg, numeric, categorical).fit(X_tr, y_tr)
    preds["scorecard"] = {cfg.seed: (sc.predict_proba(X_val), sc.predict_proba(X_te))}
    fitted["scorecard"] = sc

    dm = DesignMatrix(numeric, categorical).fit(X_tr)
    M_tr, M_val, M_te = dm.transform(X_tr), dm.transform(X_val), dm.transform(X_te)
    for name, fitter in FITTERS.items():
        preds[name] = {}
        for seed in range(cfg.seed, cfg.seed + cfg.n_seeds):
            t0 = time.time()
            m = fitter(cfg, seed, M_tr, y_tr, M_val, y_val)
            preds[name][seed] = (_proba(m, M_val), _proba(m, M_te))
            print(f"  {name} seed {seed}: {time.time() - t0:.1f}s")
            if seed == cfg.seed:
                fitted[name] = m
    return preds, fitted, dm


def _xgb_importance_grouped(model, dm: DesignMatrix, numeric, categorical):
    """Gain importance summed back onto original features (one-hot groups)."""
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    counts = booster.get_score(importance_type="weight")
    # total gain = mean gain * split count, per design column
    total = {c: gain.get(c, 0.0) * counts.get(c, 0.0) for c in dm.columns_}

    def origin(col):
        if col in numeric:
            return col
        for f in sorted(categorical, key=len, reverse=True):
            if col.startswith(f + "_"):
                return f
        return col

    agg = {}
    for col, g in total.items():
        agg[origin(col)] = agg.get(origin(col), 0.0) + g
    s = pd.Series(agg)
    return (s / s.sum()).sort_values(ascending=False)


def run_lending_club(cfg: Config = CFG) -> dict:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    t0 = time.time()

    print("Loading Lending Club ...")
    df = load_lending_club(cfg)
    train, val, test = split_by_vintage(df, cfg)
    y_tr, y_val, y_te = (train["target"].to_numpy(), val["target"].to_numpy(),
                         test["target"].to_numpy())
    print(f"  train {len(train):,} (bad {y_tr.mean():.3f}) | "
          f"val {len(val):,} (bad {y_val.mean():.3f}) | "
          f"test {len(test):,} (bad {y_te.mean():.3f})")

    preds, fitted, dm = _fit_all(cfg, NUMERIC, CATEGORICAL,
                                 train[FEATURES], y_tr,
                                 val[FEATURES], y_val, test[FEATURES])

    # ---- leaderboard ------------------------------------------------------
    per_run, agg = leaderboard(preds, y_val, y_te, cfg)
    per_run.to_csv(RESULTS / "leaderboard_lc_runs.csv", index=False)
    agg.to_csv(RESULTS / "leaderboard_lc.csv", index=False)

    # ---- significance (first-seed predictions) ----------------------------
    p_sc = preds["scorecard"][cfg.seed][1]
    sig_rows = []
    for name in FITTERS:
        p_ml = preds[name][cfg.seed][1]
        d_auc, p_delong = delong_test(y_te, p_ml, p_sc)
        lo, hi = bootstrap_ci(y_te, p_ml, p_sc, auc,
                              n_boot=cfg.n_bootstrap, seed=cfg.seed)
        # Brier compared after isotonic recalibration of both models
        p_ml_iso = calibrate("isotonic", preds[name][cfg.seed][0], y_val, p_ml)
        p_sc_iso = calibrate("isotonic", preds["scorecard"][cfg.seed][0], y_val, p_sc)
        blo, bhi = bootstrap_ci(y_te, p_ml_iso, p_sc_iso, brier,
                                n_boot=cfg.n_bootstrap, seed=cfg.seed)
        sig_rows.append(dict(model=name, auc_diff_vs_scorecard=d_auc,
                             delong_p=p_delong, auc_diff_ci_lo=lo,
                             auc_diff_ci_hi=hi, brier_iso_diff_ci_lo=blo,
                             brier_iso_diff_ci_hi=bhi))
    sig = pd.DataFrame(sig_rows)
    sig.to_csv(RESULTS / "significance_lc.csv", index=False)

    # ---- profit analysis (isotonic-calibrated PDs) -------------------------
    pnl = loan_pnl(y_te, test["loan_amnt"], test["installment"], cfg.lgd)
    curves, ops = [], []
    for name in preds:
        p_iso = calibrate("isotonic", preds[name][cfg.seed][0], y_val,
                          preds[name][cfg.seed][1])
        c = profit_curve(p_iso, pnl, cfg)
        c.insert(0, "model", name)
        curves.append(c)
        best = best_operating_point(c)
        ops.append(dict(model=name, cutoff=best["cutoff"],
                        accept_rate=best["accept_rate"],
                        profit_per_1000=best["profit_per_1000"]))
    curves = pd.concat(curves, ignore_index=True)
    accept_all = float(pnl.sum() / len(pnl) * 1000.0)
    ops.append(dict(model="accept_all", cutoff=1.0, accept_rate=1.0,
                    profit_per_1000=accept_all))
    curves.to_csv(RESULTS / "profit_curves_lc.csv", index=False)
    pd.DataFrame(ops).to_csv(RESULTS / "operating_points_lc.csv", index=False)

    # ---- interpretability: XGBoost importance vs scorecard IV -------------
    sc = fitted["scorecard"]
    iv = sc.iv_table().set_index("feature")["iv"]
    xgb_imp = _xgb_importance_grouped(fitted["xgboost"], dm, NUMERIC, CATEGORICAL)
    imp = (pd.DataFrame({"xgb_importance": xgb_imp, "iv": iv})
           .fillna(0.0).rename_axis("feature").reset_index())
    imp.to_csv(RESULTS / "importance_lc.csv", index=False)
    spearman = float(imp["xgb_importance"].corr(imp["iv"], method="spearman"))

    # ---- figures -----------------------------------------------------------
    rel = {}
    for name in preds:
        p_raw = preds[name][cfg.seed][1]
        p_iso = calibrate("isotonic", preds[name][cfg.seed][0], y_val, p_raw)
        rel[name] = {"raw": reliability_curve(y_te, p_raw, cfg.ece_bins),
                     "isotonic": reliability_curve(y_te, p_iso, cfg.ece_bins)}
    figures.fig_reliability(rel, FIGURES / "fig_reliability.png")
    figures.fig_profit(curves, accept_all, FIGURES / "fig_profit.png")
    figures.fig_score_dist(sc.score(test[FEATURES]), y_te,
                           FIGURES / "fig_score_dist.png")
    figures.fig_importance(imp.sort_values("xgb_importance", ascending=False)
                           .head(12), FIGURES / "fig_importance.png")

    summary = dict(
        n_train=int(len(train)), n_val=int(len(val)), n_test=int(len(test)),
        bad_rate_train=float(y_tr.mean()), bad_rate_val=float(y_val.mean()),
        bad_rate_test=float(y_te.mean()),
        scorecard_features=sc.features_,
        spearman_importance_vs_iv=spearman,
        accept_all_profit_per_1000=accept_all,
        runtime_s=round(time.time() - t0, 1),
    )
    (RESULTS / "summary_lc.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return dict(leaderboard=agg, significance=sig, summary=summary)


def run_taiwan(cfg: Config = CFG) -> dict:
    """Robustness: same protocol on the UCI Taiwan dataset, stratified split."""
    from sklearn.model_selection import train_test_split

    RESULTS.mkdir(exist_ok=True)
    print("Loading UCI Taiwan ...")
    df = load_taiwan()
    numeric = [c for c in df.columns if c not in TAIWAN_CATEGORICAL + ["target"]]
    X, y = df[numeric + TAIWAN_CATEGORICAL], df["target"].to_numpy()

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=cfg.seed)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=cfg.seed)

    preds, _, _ = _fit_all(cfg, numeric, TAIWAN_CATEGORICAL,
                           X_tr, y_tr, X_val, y_val, X_te)
    per_run, agg = leaderboard(preds, y_val, y_te, cfg)
    per_run.to_csv(RESULTS / "leaderboard_taiwan_runs.csv", index=False)
    agg.to_csv(RESULTS / "leaderboard_taiwan.csv", index=False)

    d_auc, p_delong = delong_test(
        y_te, preds["xgboost"][cfg.seed][1], preds["scorecard"][cfg.seed][1])
    sig = dict(auc_diff_xgb_vs_scorecard=d_auc, delong_p=p_delong,
               n_test=int(len(y_te)), bad_rate_test=float(y_te.mean()))
    (RESULTS / "summary_taiwan.json").write_text(json.dumps(sig, indent=2))
    print(json.dumps(sig, indent=2))
    return dict(leaderboard=agg, significance=sig)


def quick_config() -> Config:
    """Small config for smoke tests."""
    return replace(CFG, n_seeds=2, n_bootstrap=100,
                   xgb_params={**CFG.xgb_params, "n_estimators": 200},
                   lgbm_params={**CFG.lgbm_params, "n_estimators": 200},
                   mlp_params={**CFG.mlp_params, "max_iter": 15})
