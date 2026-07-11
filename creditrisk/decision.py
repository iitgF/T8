"""Expected-profit accept/reject analysis.

The lender accepts every applicant whose predicted PD is at or below a cutoff.
Realized profit on the test cohort, per loan i (undiscounted, stated in the
report):

    good loan:  profit_i = 36 * installment_i - loan_amnt_i   (interest earned)
    bad  loan:  profit_i = -LGD * loan_amnt_i                 (principal loss)

Profit is reported per 1,000 applicants so cohorts of different size are
comparable. Sweeping the cutoff traces the profit curve; its maximum is the
operating point a profit-seeking lender would choose with that model.
"""

import numpy as np
import pandas as pd

from .config import Config


def loan_pnl(y, loan_amnt, installment, lgd, n_payments=36):
    """Realized profit per loan given the observed outcome."""
    y = np.asarray(y)
    loan_amnt = np.asarray(loan_amnt, dtype=float)
    installment = np.asarray(installment, dtype=float)
    interest = n_payments * installment - loan_amnt
    return np.where(y == 0, interest, -lgd * loan_amnt)


def profit_curve(p_hat, pnl, cfg: Config) -> pd.DataFrame:
    """Profit per 1,000 applicants at each PD cutoff, plus acceptance rate."""
    p_hat = np.asarray(p_hat)
    n = len(p_hat)
    cutoffs = np.linspace(0.01, 0.99, cfg.n_cutoffs)
    rows = []
    for c in cutoffs:
        accept = p_hat <= c
        profit = pnl[accept].sum() / n * 1000.0
        rows.append((c, accept.mean(), profit))
    return pd.DataFrame(rows, columns=["cutoff", "accept_rate", "profit_per_1000"])


def best_operating_point(curve: pd.DataFrame) -> pd.Series:
    return curve.loc[curve["profit_per_1000"].idxmax()]
