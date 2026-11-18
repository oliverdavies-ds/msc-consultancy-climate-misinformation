"""Tests for the data quality checks (corpus/checks.py)."""

import pandas as pd
import pytest

from corpus.checks import check_corpus, find_problems

GOOD_ROWS = [
    ["GB1_0", "GB News", "2024-03-22", "Net zero targets will cost every household in Britain thousands of pounds a year."],
    ["DS4_2", "Daily Sceptic", "2025-01-10", "Independent analysts say the cost of reaching net zero has been overstated by critics."],
]
COLUMNS = ["statement_id", "source", "date_published", "statement"]


def make_table(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


def test_clean_table_passes():
    assert find_problems(make_table(GOOD_ROWS)) == []
    check_corpus(make_table(GOOD_ROWS))  # no error


def test_finds_duplicate_ids_and_statements():
    problems = find_problems(make_table(GOOD_ROWS + [GOOD_ROWS[0]]))
    assert "duplicate statement IDs: 1" in problems
    assert "duplicate statements: 1" in problems


def test_finds_missing_values():
    rows = [GOOD_ROWS[0], ["DS4_2", None, "2025-01-10", "   "]]
    problems = find_problems(make_table(rows))
    assert "missing source: 1" in problems
    assert "missing statement: 1" in problems


def test_finds_dates_outside_study_period():
    rows = [GOOD_ROWS[0], ["DS4_2", "Daily Sceptic", "2019-05-01", GOOD_ROWS[1][3]]]
    with pytest.raises(ValueError, match="dates outside"):
        check_corpus(make_table(rows))
