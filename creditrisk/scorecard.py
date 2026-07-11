"""Industry-style scorecard: IV screening -> WoE transform -> logistic
regression -> points-to-double-odds scaling.

This is deliberately the regulated-bank workhorse (Siddiqi, 2006): every input
enters through a monotone-ish WoE transform, the model is a plain logistic
regression, and the output can be expressed both as a PD and as an additive
integer score that a loan officer can read line by line.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .config import Config
from .woe import WoEBinner


class Scorecard:
    def __init__(self, cfg: Config, numeric, categorical):
        self.cfg = cfg
        self.binner = WoEBinner(numeric, categorical,
                                max_bins=cfg.woe_max_bins,
                                min_frac=cfg.woe_min_frac)
        self.lr = LogisticRegression(C=1.0, max_iter=1000)
        self.features_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Scorecard":
        self.binner.fit(X, y)
        iv = self.binner.iv_table()
        self.features_ = iv[iv["iv"] >= self.cfg.iv_threshold]["feature"].tolist()
        W = self.binner.transform(X)[self.features_]
        self.lr.fit(W, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        W = self.binner.transform(X)[self.features_]
        return self.lr.predict_proba(W)[:, 1]

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Additive score: higher = safer. base_score at base_odds good:bad,
        +pdo points doubles the odds of being good."""
        pd_hat = self.predict_proba(X)
        odds_good = (1 - pd_hat) / pd_hat
        factor = self.cfg.pdo / np.log(2)
        offset = self.cfg.base_score - factor * np.log(self.cfg.base_odds)
        return offset + factor * np.log(odds_good)

    def iv_table(self) -> pd.DataFrame:
        return self.binner.iv_table()
