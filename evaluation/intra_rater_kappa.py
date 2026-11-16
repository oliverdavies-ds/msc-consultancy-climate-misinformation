"""
Intra rater reliability of the hand labelled testbed.

The 1,002 statements were labelled twice, four weeks apart. Cohen's kappa
between the two rounds is reported for the full code set of each statement and
at super claim level.
"""

import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "public" / "testbed_adjudication.csv")


def codes(cell):
    """All CARDS codes in a cell, deduplicated and sorted."""
    return sorted(set(re.findall(r"\d+(?:_\d+)+", str(cell))))


def full_label(cell):
    """The complete set of codes as one label, so multi label rows must match exactly."""
    return ",".join(codes(cell))


def super_label(cell):
    """The set of super claims only (the first level of each code)."""
    return ",".join(sorted(set(code.split("_")[0] for code in codes(cell))))


print("Full kappa:", cohen_kappa_score(df["round_1"].apply(full_label), df["round_2"].apply(full_label)))
print("Super kappa:", cohen_kappa_score(df["round_1"].apply(super_label), df["round_2"].apply(super_label)))
