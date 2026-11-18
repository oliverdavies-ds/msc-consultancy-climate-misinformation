"""
Merge the filtered statements from all three source groups into one corpus.

Input:  data/interim/{source}_statements_climate.jsonl
Output: data/corpus/corpus.csv
"""

import json
from pathlib import Path

import pandas as pd

from checks import check_corpus

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
OUT_PATH = ROOT / "data" / "corpus" / "corpus.csv"

SOURCES = ["gbnews", "daily_sceptic", "think_tanks"]
COLUMNS = ["statement_id", "source", "date_published", "title", "statement"]

rows = [json.loads(line)
        for source in SOURCES
        for line in open(INTERIM / f"{source}_statements_climate.jsonl", encoding="utf-8")]

# Identical statements can recur across articles (syndicated copy, repeated
# straplines), so only the first occurrence is kept.
corpus = pd.DataFrame(rows)[COLUMNS].drop_duplicates(subset="statement")

# Stops here, before saving, if the corpus fails any data quality check.
check_corpus(corpus)
corpus.to_csv(OUT_PATH, index=False)
