"""ML challengers on a shared design matrix.

All three challengers (XGBoost, LightGBM, MLP) consume the same design matrix
(median-imputed numerics + one-hot categoricals, fit on train only), so
differences in performance come from the learner, not the preprocessing.
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .config import Config


class DesignMatrix:
    """Median impute numerics, one-hot categoricals. Fit on train only."""

    def __init__(self, numeric, categorical):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.medians_ = None
        self.columns_ = None

    def fit(self, X: pd.DataFrame) -> "DesignMatrix":
        self.medians_ = X[self.numeric].median()
        self.columns_ = self._encode(X).columns
        return self

    def _encode(self, X: pd.DataFrame) -> pd.DataFrame:
        num = X[self.numeric].fillna(self.medians_)
        cat = pd.get_dummies(X[self.categorical].astype(str), dtype=float)
        return pd.concat([num, cat], axis=1)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        M = self._encode(X)
        return M.reindex(columns=self.columns_, fill_value=0.0).astype(float)


def fit_xgboost(cfg: Config, seed, X_tr, y_tr, X_val, y_val):
    from xgboost import XGBClassifier
    model = XGBClassifier(random_state=seed, **cfg.xgb_params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def fit_lightgbm(cfg: Config, seed, X_tr, y_tr, X_val, y_val):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(random_state=seed, **cfg.lgbm_params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    return model


class MLPWrapper:
    """MLP with standardized inputs; sklearn handles early stopping."""

    def __init__(self, cfg: Config, seed):
        self.scaler = StandardScaler()
        self.mlp = MLPClassifier(random_state=seed, **cfg.mlp_params)

    def fit(self, X_tr, y_tr):
        Z = self.scaler.fit_transform(X_tr)
        self.mlp.fit(Z, y_tr)
        return self

    def predict_proba(self, X):
        return self.mlp.predict_proba(self.scaler.transform(X))


def fit_mlp(cfg: Config, seed, X_tr, y_tr, X_val=None, y_val=None):
    return MLPWrapper(cfg, seed).fit(X_tr, y_tr)
