# Does Machine Learning Beat the Credit Scorecard?

Term project (B.Sc. Hons Data Science & AI, IIT Guwahati, Trimester 8) testing whether
the machine-learning advantage over the industry-standard logistic scorecard survives
the criteria a lender actually needs: not just discrimination (AUC), but calibration
and accept/reject profit, under a strict out-of-time protocol with no leakage.

**Headline result:** on 621,022 Lending Club 36-month loans (train ≤2013, calibrate on
2014, test on 2015), XGBoost beats a weight-of-evidence scorecard by **0.85 AUC points**
(DeLong p < 1e-54) and the edge survives isotonic recalibration and converts into
**+2.1% profit** at the optimal cutoff. But the margins are modest, the MLP is
*significantly worse* than the scorecard, and neither tree ensemble matches the
scorecard's calibration. The ranking reproduces on the UCI Taiwan dataset.

## Leaderboard (out-of-time test = 2015 vintage, 283,026 loans)

| Calib. | Model | AUC | KS | Brier | ECE |
|---|---|---|---|---|---|
| Raw | Scorecard | 0.6767 | 0.2570 | 0.1213 | 0.0246 |
| | XGBoost | **0.6850** | **0.2701** | 0.1212 | 0.0338 |
| | LightGBM | 0.6843 | 0.2693 | 0.1213 | 0.0344 |
| | MLP | 0.6716 | 0.2542 | 0.1218 | 0.0275 |
| Isotonic | Scorecard | 0.6766 | 0.2564 | 0.1209 | 0.0175 |
| | XGBoost | 0.6848 | 0.2699 | **0.1203** | 0.0214 |
| | LightGBM | 0.6841 | 0.2688 | 0.1204 | 0.0218 |
| | MLP | 0.6714 | 0.2537 | 0.1212 | **0.0172** |

Best profit policy: XGBoost rejects the riskiest 9.3% of applicants for +3.3% profit
versus accepting everyone; the scorecard's best policy gains +1.2%.

## Repository layout

```
creditrisk/                  modular pipeline (importable package)
  config.py                  run parameters, vintage split, profit assumptions
  data.py                    Lending Club / UCI Taiwan loaders + leakage whitelist
  woe.py                     weight-of-evidence binning and information value
  scorecard.py               WoE -> IV screen -> logistic -> points scaling
  models.py                  XGBoost, LightGBM, MLP on a shared design matrix
  calibration.py             Platt and isotonic recalibration (fit on validation)
  metrics.py                 AUC, KS, Brier, ECE, DeLong test, bootstrap CIs
  decision.py                accept/reject profit model and cutoff sweep
  evaluate.py                leaderboard assembly
  figures.py                 report figures
  pipeline.py                end-to-end orchestration

run_study.py                 entry point (python run_study.py [--quick])
study_notebook.ipynb         annotated study over the saved results
inspect_csv.py               summary of the raw export: rows, size, kept vs
                             dropped columns, then N rows with all headers
figures/                     figures used by the report
results/                     leaderboards, significance tests, profit curves
data/download_lc.py          re-downloads the Lending Club file (374 MB)

report_and_video/            written and video deliverables
  report.tex                 term-project report (compile on Overleaf)
  report_simple.tex          condensed short version of the report
  T8.pdf                     compiled report (short, memos, deck also here)
  memo_standard.tex          concepts and methodology, with sample Q&A
  memo_condensed.tex         two-page rapid reference
  voiceover.tex              per-slide narration script for the video
  make_deck.py               builds the 12-slide deck from the figures
  Project_Video_Presentation.pptx   the deck
  title_slide.png            IITG title slide, also the YouTube thumbnail
```

## Reproducing

```bash
pip install numpy pandas scipy scikit-learn xgboost lightgbm matplotlib xlrd kagglehub
python data/download_lc.py     # Lending Club accepted loans (anonymous, 374 MB)
# UCI Taiwan: https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip
#   -> unzip into data/
python run_study.py            # full study, ~18 min on a laptop CPU
```

All seeds are fixed (42-46 for the five ML seeds) and every model shares a single
vintage split, so results are reproducible. Tested with Python 3.13, numpy 2.4,
pandas 3.0, scikit-learn 1.9, xgboost 3.3, lightgbm 4.6.

## Note on AI assistance

AI-assisted tooling (Claude Code) was used for code refactoring, optimization, and
drafting support. All modelling decisions, verification of results, and the final text
are the author's responsibility, and all reported numbers were reproduced end-to-end
from the committed code.
