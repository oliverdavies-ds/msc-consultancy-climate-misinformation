"""
Descriptive statistics for the corpus before and after ClimateBERT filtering
(Tables 1 and 2 of the dissertation).
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
CORPUS_PATH = ROOT / "data" / "corpus" / "corpus.csv"

SOURCES = ["daily_sceptic", "gbnews", "think_tanks"]
OUTLETS = ["GB News", "Daily Sceptic"]
STATS = ["count", "mean", "median", "std", "sum"]


def add_group_and_word_count(df):
    """Pool the five think tanks into one group and count the words in each statement."""
    df["group"] = df["source"].where(df["source"].isin(OUTLETS), "Think tanks")
    df["word_count"] = df["statement"].apply(lambda x: len(str(x).split()))
    return df


#%% Table 1: all statements before filtering

unfiltered = {s: pd.read_json(INTERIM / f"{s}_statements.jsonl", lines=True) for s in SOURCES}
before = add_group_and_word_count(pd.concat(unfiltered.values(), ignore_index=True))

pre_filter = before.groupby("group")["word_count"].agg(STATS).round(2)
print(pre_filter)
print(before["word_count"].agg(["mean", "median", "std", "sum"]))

#%% Table 2: final corpus after filtering and deduplication

after = add_group_and_word_count(pd.read_csv(CORPUS_PATH))

post_filter = after.groupby("group")["word_count"].agg(STATS).round(2)
print(post_filter)
print(after["word_count"].agg(["mean", "median", "std", "sum"]))

#%% Share of statements ClimateBERT retained, by source

for source in SOURCES:
    dropped = pd.read_json(INTERIM / f"{source}_statements_dropped.jsonl", lines=True)
    print(source, "retained:", round(1 - len(dropped) / len(unfiltered[source]), 4))
