"""
Data quality checks for the corpus.

check_corpus() prints a short report and stops with an error if anything is
wrong, so merge_corpus.py never saves a bad corpus. It also runs on any saved
table of statements, including the label only file in data/public/:

    python corpus/checks.py data/corpus/corpus.csv
    python corpus/checks.py data/public/corpus_results.csv
"""

import sys

import pandas as pd

REQUIRED_COLUMNS = ["statement_id", "source", "date_published"]
TEXT_COLUMN = "statement"  # checked only when the table includes the text
MIN_WORDS = 10
MAX_WORDS = 300
START_DATE = "2023-09-01"  # study period
END_DATE = "2026-06-30"


def find_problems(df):
    """Return a list of problems in a table of statements. An empty list means it passed."""
    absent = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if absent:
        return [f"missing columns {absent}"]

    problems = []

    # Missing or blank values
    columns = REQUIRED_COLUMNS + ([TEXT_COLUMN] if TEXT_COLUMN in df.columns else [])
    for col in columns:
        blank = df[col].isna() | (df[col].astype(str).str.strip() == "")
        if blank.any():
            problems.append(f"missing {col}: {blank.sum()}")

    # Duplicates
    duplicate_ids = df["statement_id"].duplicated().sum()
    if duplicate_ids:
        problems.append(f"duplicate statement IDs: {duplicate_ids}")

    if TEXT_COLUMN in df.columns:
        duplicate_text = df[TEXT_COLUMN].duplicated().sum()
        if duplicate_text:
            problems.append(f"duplicate statements: {duplicate_text}")

        # Statement length should match the limits used in split_statements.py
        words = df[TEXT_COLUMN].fillna("").astype(str).str.split().str.len()
        wrong_length = ((words < MIN_WORDS) | (words > MAX_WORDS)).sum()
        if wrong_length:
            problems.append(f"statements outside {MIN_WORDS} to {MAX_WORDS} words: {wrong_length}")

    # Dates must be readable and inside the study period
    dates = pd.to_datetime(df["date_published"], errors="coerce")
    unreadable = (dates.isna() & df["date_published"].notna()).sum()
    if unreadable:
        problems.append(f"unreadable dates: {unreadable}")
    outside = ((dates < START_DATE) | (dates > END_DATE)).sum()
    if outside:
        problems.append(f"dates outside {START_DATE} to {END_DATE}: {outside}")

    return problems


def report(df):
    """Print row counts by source and the date range."""
    print(f"Rows: {len(df):,}")
    for source, n in df["source"].value_counts().items():
        print(f"  {source}: {n:,}")
    dates = pd.to_datetime(df["date_published"], errors="coerce")
    if dates.notna().any():
        print(f"Dates: {dates.min():%d %b %Y} to {dates.max():%d %b %Y}")


def check_corpus(df):
    """Print the report, then stop with an error if any check fails."""
    problems = find_problems(df)
    if all(col in df.columns for col in REQUIRED_COLUMNS):
        report(df)
    if problems:
        raise ValueError("Data quality checks failed:\n  " + "\n  ".join(problems))
    print("All checks passed")


if __name__ == "__main__":
    check_corpus(pd.read_csv(sys.argv[1]))
