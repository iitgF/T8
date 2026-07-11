"""Leaderboard assembly: every model x calibration x seed, then aggregates."""

import numpy as np
import pandas as pd

from .calibration import calibrate
from .config import Config
from .metrics import auc, brier, ece, ks_stat


def leaderboard(preds: dict, y_val, y_test, cfg: Config):
    """preds: {model: {seed: (p_val, p_test)}} raw model probabilities.

    Returns (per_run, aggregated) DataFrames. Calibrators are fit on the
    validation split for each individual run, never on test data.
    """
    rows = []
    for model, seed_preds in preds.items():
        for seed, (p_val, p_test) in seed_preds.items():
            for calib in ["raw", "platt", "isotonic"]:
                p = calibrate(calib, p_val, y_val, p_test)
                rows.append(dict(
                    model=model, seed=seed, calibration=calib,
                    auc=auc(y_test, p), ks=ks_stat(y_test, p),
                    brier=brier(y_test, p), ece=ece(y_test, p, cfg.ece_bins),
                ))
    per_run = pd.DataFrame(rows)
    agg = (per_run
           .groupby(["model", "calibration"], as_index=False)
           .agg(auc_mean=("auc", "mean"), auc_std=("auc", "std"),
                ks_mean=("ks", "mean"), brier_mean=("brier", "mean"),
                brier_std=("brier", "std"), ece_mean=("ece", "mean"),
                ece_std=("ece", "std")))
    return per_run, agg
