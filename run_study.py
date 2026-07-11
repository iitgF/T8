"""Run the full study: Lending Club main experiment + Taiwan robustness check.

Usage:
    python run_study.py            # full run
    python run_study.py --quick    # fast smoke test (reduced models/bootstrap)
"""

import argparse

from creditrisk.config import CFG
from creditrisk.pipeline import quick_config, run_lending_club, run_taiwan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke test")
    ap.add_argument("--skip-taiwan", action="store_true")
    args = ap.parse_args()

    cfg = quick_config() if args.quick else CFG

    out = run_lending_club(cfg)
    print("\n=== Lending Club leaderboard ===")
    print(out["leaderboard"].round(4).to_string(index=False))

    if not args.skip_taiwan:
        out_tw = run_taiwan(cfg)
        print("\n=== Taiwan leaderboard ===")
        print(out_tw["leaderboard"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
