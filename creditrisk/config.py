"""Run parameters shared by every experiment.

A single frozen config object keeps every model on the same data, the same
split and the same seeds, so results are comparable and reproducible.
"""

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LC_CSV = (
    ROOT
    / "data"
    / "_kagglehub"
    / "datasets"
    / "wordsforthewise"
    / "lending-club"
    / "versions"
    / "3"
    / "accepted_2007_to_2018Q4.csv.gz"
)
TAIWAN_XLS = ROOT / "data" / "default of credit card clients.xls"


@dataclass(frozen=True)
class Config:
    # reproducibility
    seed: int = 42
    n_seeds: int = 5              # ML models retrained with seeds seed..seed+n_seeds-1

    # Lending Club vintage split (36-month loans only, so every loan matures
    # before the data ends in 2018Q4 and outcomes are fully observed)
    train_end: str = "2013-12-31"   # issue date <= -> train
    val_end: str = "2014-12-31"     # issue date <= -> validation (calibration)
    test_end: str = "2015-12-31"    # issue date <= -> out-of-time test

    # scorecard
    woe_max_bins: int = 8
    woe_min_frac: float = 0.05      # minimum fraction of train rows per bin
    iv_threshold: float = 0.02      # drop features with IV below this
    pdo: float = 20.0               # points to double the odds
    base_score: float = 600.0       # score at base odds
    base_odds: float = 19.0         # good:bad odds at base_score

    # ML models
    xgb_params: dict = field(default_factory=lambda: dict(
        n_estimators=2000, learning_rate=0.05, max_depth=5,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", early_stopping_rounds=100,
        eval_metric="auc", n_jobs=-1,
    ))
    lgbm_params: dict = field(default_factory=lambda: dict(
        n_estimators=2000, learning_rate=0.05, num_leaves=31,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, verbosity=-1,
    ))
    mlp_params: dict = field(default_factory=lambda: dict(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
        alpha=1e-4, batch_size=512, learning_rate_init=1e-3,
        max_iter=60, early_stopping=True, n_iter_no_change=5,
    ))

    # profit model (undiscounted, stated assumptions; see report Sec. 3)
    lgd: float = 0.65               # loss given default on the principal
    n_cutoffs: int = 99             # PD cutoffs swept for the profit curve

    # evaluation
    n_bootstrap: int = 1000
    ece_bins: int = 10


CFG = Config()
