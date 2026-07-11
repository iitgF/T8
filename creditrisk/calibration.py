"""Post-hoc probability calibration, fit on the out-of-time validation split.

Platt scaling refits a one-dimensional logistic regression on the raw model
logit; isotonic regression fits a monotone step function on the raw
probability. Fitting on the 2014 vintage and testing on 2015 mimics how a
bank would recalibrate a deployed model on its most recent observed cohort.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class PlattCalibrator:
    def fit(self, p_val, y_val):
        self.lr = LogisticRegression(C=1e6, max_iter=1000)
        self.lr.fit(_logit(p_val).reshape(-1, 1), y_val)
        return self

    def transform(self, p):
        return self.lr.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    def fit(self, p_val, y_val):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS)
        self.iso.fit(p_val, y_val)
        return self

    def transform(self, p):
        return self.iso.predict(p)


CALIBRATORS = {"raw": None, "platt": PlattCalibrator, "isotonic": IsotonicCalibrator}


def calibrate(name, p_val, y_val, p_test):
    """Return test-set probabilities after the named calibration step."""
    if name == "raw":
        return p_test
    cal = CALIBRATORS[name]().fit(p_val, y_val)
    return cal.transform(p_test)
