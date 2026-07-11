"""Weight-of-evidence binning and information value.

WoE for bin j:  WoE_j = ln( (goods_j / goods_total) / (bads_j / bads_total) )
IV for feature: IV = sum_j (p_good_j - p_bad_j) * WoE_j

Numeric features are quantile-binned on the training data (missing values get
their own bin); categorical features use their levels, with rare levels pooled
into an OTHER bin. Counts are smoothed by 0.5 so empty cells never produce
infinite WoE. Everything is fit on the training split only.
"""

import numpy as np
import pandas as pd

MISSING = "__MISSING__"
OTHER = "__OTHER__"


class WoEBinner:
    """Fits per-feature WoE maps on train data, transforms any split."""

    def __init__(self, numeric, categorical, max_bins=8, min_frac=0.05):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.max_bins = max_bins
        self.min_frac = min_frac
        self.edges_ = {}       # feature -> np.array of bin edges
        self.woe_maps_ = {}    # feature -> {bin label: woe}
        self.iv_ = {}          # feature -> information value

    # -- helpers ------------------------------------------------------------

    def _bin_labels_numeric(self, x: pd.Series, edges: np.ndarray) -> pd.Series:
        lab = pd.cut(x, bins=edges, include_lowest=True).astype(str)
        return lab.where(x.notna(), MISSING)

    def _woe_iv(self, labels: pd.Series, y: pd.Series):
        tab = pd.crosstab(labels, y)
        goods = tab.get(0, pd.Series(0, index=tab.index)) + 0.5
        bads = tab.get(1, pd.Series(0, index=tab.index)) + 0.5
        pg, pb = goods / goods.sum(), bads / bads.sum()
        woe = np.log(pg / pb)
        iv = float(((pg - pb) * woe).sum())
        return woe.to_dict(), iv

    # -- API ----------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEBinner":
        n = len(X)
        for f in self.numeric:
            x = X[f]
            # quantile bins with at least min_frac of rows each
            n_bins = min(self.max_bins, max(2, int(1 / self.min_frac)))
            edges = np.unique(np.nanquantile(x, np.linspace(0, 1, n_bins + 1)))
            edges[0], edges[-1] = -np.inf, np.inf
            if len(edges) < 3:                       # degenerate feature
                edges = np.array([-np.inf, np.inf])
            self.edges_[f] = edges
            labels = self._bin_labels_numeric(x, edges)
            self.woe_maps_[f], self.iv_[f] = self._woe_iv(labels, y)

        for f in self.categorical:
            x = X[f].fillna(MISSING)
            counts = x.value_counts(normalize=True)
            rare = set(counts[counts < 0.02].index)
            labels = x.where(~x.isin(rare), OTHER)
            self.woe_maps_[f], self.iv_[f] = self._woe_iv(labels, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for f in self.numeric:
            labels = self._bin_labels_numeric(X[f], self.edges_[f])
            out[f] = labels.map(self.woe_maps_[f]).fillna(0.0)
        for f in self.categorical:
            x = X[f].fillna(MISSING)
            known = set(self.woe_maps_[f])
            labels = x.where(x.isin(known), OTHER)
            out[f] = labels.map(self.woe_maps_[f]).fillna(0.0)
        return pd.DataFrame(out, index=X.index).astype(float)

    def iv_table(self) -> pd.DataFrame:
        return (pd.DataFrame({"feature": list(self.iv_), "iv": list(self.iv_.values())})
                .sort_values("iv", ascending=False)
                .reset_index(drop=True))
