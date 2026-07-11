"""Data loading, leakage filtering and the out-of-time vintage split.

The central discipline of this module is the leakage filter: the raw Lending
Club export has 151 columns, most of which are only observable *after*
origination (payments received, recoveries, hardship flags, ...). A PD model
must only see what the lender knew on the day the loan was issued, so we keep
an explicit whitelist of origination-time fields and derive everything else
from them.
"""

import numpy as np
import pandas as pd

from .config import Config, LC_CSV, TAIWAN_XLS

# ---------------------------------------------------------------------------
# Lending Club
# ---------------------------------------------------------------------------

# Origination-time whitelist. Everything not listed here is dropped unread.
LC_RAW_COLS = [
    # loan terms set at origination
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    # applicant profile at application time
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "purpose", "dti", "application_type",
    # credit bureau snapshot pulled at application time
    "fico_range_low", "fico_range_high", "earliest_cr_line", "delinq_2yrs",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "mort_acc", "pub_rec_bankruptcies",
    # bookkeeping (used for the split / target only, never as features)
    "issue_d", "loan_status",
]

# Final resolved statuses. Loans still in flight are excluded so the target
# is fully observed for every retained loan.
GOOD_STATUS = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}
BAD_STATUS = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
}

CATEGORICAL = ["grade", "sub_grade", "home_ownership", "purpose",
               "verification_status", "application_type"]

NUMERIC = ["loan_amnt", "int_rate", "installment", "annual_inc", "dti",
           "fico", "emp_length_yrs", "credit_hist_yrs", "delinq_2yrs",
           "inq_last_6mths", "open_acc", "pub_rec", "revol_bal",
           "revol_util", "total_acc", "mort_acc", "pub_rec_bankruptcies"]

FEATURES = NUMERIC + CATEGORICAL


def _parse_emp_length(s: pd.Series) -> pd.Series:
    """'10+ years' -> 10, '< 1 year' -> 0, '3 years' -> 3, missing -> NaN."""
    out = (
        s.str.replace("10+ years", "10", regex=False)
         .str.replace("< 1 year", "0", regex=False)
         .str.extract(r"(\d+)")[0]
    )
    return pd.to_numeric(out, errors="coerce")


def load_lending_club(cfg: Config) -> pd.DataFrame:
    """Load, filter and feature-engineer the Lending Club accepted loans.

    Returns one row per 36-month loan issued 2007-2015 with a resolved
    outcome, columns = FEATURES + [target, issue_d, loan_amnt, installment].
    """
    df = pd.read_csv(LC_CSV, usecols=LC_RAW_COLS, low_memory=False)

    # 36-month loans only: every loan issued up to Dec-2015 has fully matured
    # by the end of the data (2018Q4), so outcomes are not right-censored.
    df = df[df["term"].str.strip() == "36 months"]

    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df = df[df["issue_d"] <= pd.Timestamp(cfg.test_end)]

    # resolved outcomes only
    df = df[df["loan_status"].isin(GOOD_STATUS | BAD_STATUS)].copy()
    df["target"] = df["loan_status"].isin(BAD_STATUS).astype(int)

    # engineered origination-time features
    df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2.0
    df["emp_length_yrs"] = _parse_emp_length(df["emp_length"])
    earliest = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")
    df["credit_hist_yrs"] = (df["issue_d"] - earliest).dt.days / 365.25

    keep = FEATURES + ["target", "issue_d"]
    return df[keep].reset_index(drop=True)


def split_by_vintage(df: pd.DataFrame, cfg: Config):
    """Out-of-time split: train <= 2013, validation = 2014, test = 2015."""
    train = df[df["issue_d"] <= pd.Timestamp(cfg.train_end)]
    val = df[(df["issue_d"] > pd.Timestamp(cfg.train_end))
             & (df["issue_d"] <= pd.Timestamp(cfg.val_end))]
    test = df[(df["issue_d"] > pd.Timestamp(cfg.val_end))
              & (df["issue_d"] <= pd.Timestamp(cfg.test_end))]
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


# ---------------------------------------------------------------------------
# UCI Taiwan credit-card default (robustness dataset)
# ---------------------------------------------------------------------------

TAIWAN_CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]


def load_taiwan() -> pd.DataFrame:
    """UCI 'default of credit card clients' (Yeh & Lien, 2009), 30k rows.

    No origination dates exist, so the study uses a stratified random split
    for this dataset (see pipeline).
    """
    df = pd.read_excel(TAIWAN_XLS, header=1)
    df = df.rename(columns={"default payment next month": "target"})
    df = df.drop(columns=["ID"])
    for c in TAIWAN_CATEGORICAL:
        df[c] = df[c].astype(str)
    return df
