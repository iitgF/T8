"""Discrimination and calibration metrics plus significance machinery.

AUC / KS measure ranking quality; Brier and ECE measure whether the predicted
PDs can be taken at face value. DeLong's test compares two correlated AUCs on
the same test set; the bootstrap gives CIs for any paired metric difference.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score


def auc(y, p):
    return float(roc_auc_score(y, p))


def ks_stat(y, p):
    """Kolmogorov-Smirnov distance between score CDFs of goods and bads."""
    y = np.asarray(y)
    p = np.asarray(p)
    res = stats.ks_2samp(p[y == 1], p[y == 0])
    return float(res.statistic)


def brier(y, p):
    return float(brier_score_loss(y, p))


def ece(y, p, n_bins=10):
    """Expected calibration error with equal-frequency bins."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    splits = np.array_split(order, n_bins)
    total = 0.0
    for idx in splits:
        if len(idx) == 0:
            continue
        total += len(idx) / len(p) * abs(y[idx].mean() - p[idx].mean())
    return float(total)


def reliability_curve(y, p, n_bins=10):
    """(mean predicted PD, observed default rate, bin count) per quantile bin."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    rows = []
    for idx in np.array_split(order, n_bins):
        rows.append((p[idx].mean(), y[idx].mean(), len(idx)))
    return pd.DataFrame(rows, columns=["p_mean", "y_rate", "n"])


# ---------------------------------------------------------------------------
# DeLong test for two correlated AUCs (Sun & Xu, 2014 fast implementation)
# ---------------------------------------------------------------------------

def _midrank(x):
    order = np.argsort(x)
    x_sorted = x[order]
    n = len(x)
    cum = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and x_sorted[j] == x_sorted[i]:
            j += 1
        cum[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = cum
    return out


def _delong_cov(y, preds):
    """preds: (k, n) array of scores; returns AUCs and covariance matrix."""
    y = np.asarray(y)
    pos = preds[:, y == 1]
    neg = preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = preds.shape[0]
    tx = np.array([_midrank(pos[i]) for i in range(k)])
    ty = np.array([_midrank(neg[i]) for i in range(k)])
    tz = np.array([_midrank(np.concatenate([pos[i], neg[i]])) for i in range(k)])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    s01 = np.cov(v01)
    s10 = np.cov(v10)
    cov = s01 / m + s10 / n
    return aucs, np.atleast_2d(cov)


def delong_test(y, p1, p2):
    """Two-sided p-value for H0: AUC(p1) == AUC(p2) on the same test set."""
    aucs, cov = _delong_cov(y, np.vstack([p1, p2]))
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return diff, 1.0
    z = diff / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))
    return float(diff), float(p)


def bootstrap_ci(y, p1, p2, fn, n_boot=1000, seed=42):
    """Percentile CI for fn(y, p1) - fn(y, p2) under test-set resampling."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    n = len(y)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = fn(y[idx], p1[idx]) - fn(y[idx], p2[idx])
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
