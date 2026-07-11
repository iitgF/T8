"""creditrisk: scorecard vs. ML study for probability-of-default modelling.

Modules
-------
config       run parameters shared by every experiment
data         Lending Club / UCI Taiwan loaders, leakage filter, vintage split
woe          weight-of-evidence binning and information value
scorecard    WoE + logistic regression scorecard with points scaling
models       XGBoost, LightGBM and MLP challengers on a shared design matrix
calibration  Platt and isotonic post-hoc calibration
metrics      AUC, KS, Brier, ECE, DeLong test, bootstrap CIs
decision     expected-profit accept/reject analysis
evaluate     leaderboard assembly
pipeline     end-to-end orchestration
"""

__version__ = "1.0.0"
