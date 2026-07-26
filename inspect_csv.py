"""Inspect a very large CSV without loading it into memory.

Prints a summary table (row count, file size, the origination-time columns
kept by the study and examples of the post-origination columns dropped),
then asks how many rows to show and prints them with every column header.
With many columns a flat table is unreadable, so the rows are shown
transposed: one line per column, the header, then the value in each row.
The untransposed sample is also written to <file>_sample.csv.

Run it any way you like:
    python inspect_csv.py                     # asks for the file
    python inspect_csv.py <file.csv[.gz]>     # file given up front
    python inspect_csv.py <file.csv> 25       # file and row count given
Double-clicking or an IDE Run button also works; if no console input is
available the defaults are used instead of crashing.
"""

import gzip
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

# The study's origination-time whitelist, imported so this script and the
# pipeline can never disagree about which columns are safe to use.
try:
    from creditrisk.data import LC_RAW_COLS as KEPT
except ImportError:                      # standalone copy of the script
    KEPT = []

# Unambiguous post-origination fields: known only after money changed hands,
# so using any of them to predict default would leak the answer.
LEAK_EXAMPLES = ["total_pymnt", "total_rec_prncp", "recoveries",
                 "collection_recovery_fee", "last_pymnt_amnt",
                 "last_fico_range_high"]


def ask(prompt, default):
    """input() that survives having no console attached."""
    try:
        answer = input(f"{prompt} [{default}]: ").strip()
    except (EOFError, OSError):
        print("(no console input, using the default)")
        return str(default)
    return answer or str(default)


def find_csv():
    """Largest .csv/.csv.gz next to this script, as the offered default."""
    files = [p for p in HERE.iterdir()
             if p.suffix == ".csv" or p.name.endswith(".csv.gz")]
    files = [p for p in files if not p.stem.endswith("_sample")]
    return max(files, key=lambda p: p.stat().st_size) if files else None


def count_rows(path):
    """Count data rows by counting newlines in 1 MB binary blocks.

    Assumes no newlines inside quoted fields, which is what makes it fast
    (about a second per gigabyte rather than a full parse).
    """
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as f:
        n = sum(block.count(b"\n") for block in iter(lambda: f.read(1 << 20), b""))
    return n - 1


def wrap(names, width=88, indent="  "):
    """Comma-separated names wrapped to a readable block."""
    lines, line = [], indent
    for i, name in enumerate(names):
        piece = name + (", " if i < len(names) - 1 else "")
        if len(line) + len(piece) > width:
            lines.append(line)
            line = indent
        line += piece
    return "\n".join(lines + [line])


def summarize(path, columns):
    kept = [c for c in KEPT if c in columns]
    leaks = [c for c in LEAK_EXAMPLES if c in columns]
    n_other = len(columns) - len(kept)

    print("\n" + "=" * 62)
    print(f"{'file':<22}{path.name}")
    print(f"{'size on disk':<22}{path.stat().st_size / 1e6:,.1f} MB")
    print(f"{'total rows':<22}{count_rows(path):,}")
    print(f"{'total columns':<22}{len(columns)}")
    print(f"{'kept (origination)':<22}{len(kept)}")
    print(f"{'dropped':<22}{n_other}")
    print("=" * 62)

    if kept:
        print(f"\nNon-leakage columns, the whole whitelist ({len(kept)}):")
        print(wrap(kept))
    else:
        print("\nNon-leakage whitelist unavailable (creditrisk package not importable).")

    if leaks:
        print(f"\nLeakage columns, {len(leaks)} examples of the {n_other} dropped:")
        print(wrap(leaks))
        print("  (known only after origination; the rest are identifiers,\n"
              "   free text, or columns that are entirely empty)")


def main():
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        default = find_csv()
        if default is None:
            print(f"No .csv file found in {HERE}")
            return
        path = Path(ask("Which file?", default.name))
    if not path.is_absolute():
        path = HERE / path
    if not path.exists():
        print(f"File not found: {path}")
        return

    header = pd.read_csv(path, nrows=0).columns.tolist()
    summarize(path, header)

    n_rows = int(sys.argv[2]) if len(sys.argv) > 2 else int(ask("\nHow many rows?", 10))
    df = pd.read_csv(path, nrows=n_rows, low_memory=False)
    print(f"\nshowing {len(df)} row(s), all {df.shape[1]} columns\n")

    view = df.T
    view.columns = [f"row {i}" for i in range(len(df))]
    view.index.name = "column"
    with pd.option_context("display.max_rows", None, "display.max_columns", None,
                           "display.width", 0, "display.max_colwidth", 30):
        print(view)

    out = path.with_name(path.stem + "_sample.csv")
    df.to_csv(out, index=False)
    print(f"\nsample also written to: {out.name}")

    if sys.stdin.isatty():           # keep a double-clicked window open
        ask("Press Enter to close", "")


if __name__ == "__main__":
    main()
