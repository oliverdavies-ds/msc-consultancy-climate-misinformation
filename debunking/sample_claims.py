"""
Draw one statement at random for every CARDS code present in the corpus, to
test the debunker across the whole taxonomy.

Note: the sample debunked in the dissertation was drawn before a seed was set.
Rerunning this script replaces it with a new, seeded sample.

Input:  data/corpus/statements_flicc.csv
Output: data/debunking/code_samples.csv
"""

import ast
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "corpus" / "statements_flicc.csv"
OUT_PATH = ROOT / "data" / "debunking" / "code_samples.csv"

SEED = 42

df = pd.read_csv(IN_PATH)


def parse(cell):
    """Turn the stored list string back into a list of codes."""
    try:
        return ast.literal_eval(cell)
    except Exception:
        return []


# One row per statement and code pair, so that a statement can be drawn for any of its codes.
df["codes"] = df["predicted_claims"].map(parse)
exploded = df.explode("codes").dropna(subset=["codes"]).rename(columns={"codes": "code"})

# Stratified sample: one statement per code.
sample = (exploded.groupby("code", group_keys=False)
          .sample(random_state=SEED)
          [["code", "statement_id", "source", "date_published", "statement", "predicted_claims", "flicc"]]
          .sort_values("code"))

sample.to_csv(OUT_PATH, index=False)
