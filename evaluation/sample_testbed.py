"""
Draw a stratified sample of 1,002 statements from the corpus for hand labelling.

Sampling is stratified by source type (media outlets against think tanks).

Note: the labelled testbed used in the dissertation was drawn before a random
seed had been set. The two labelling rounds were completed four weeks apart, so
that original sample was kept and is not reproduced by rerunning this script.
A seed is fixed here for any future draw.

Input:  data/corpus/corpus.csv
Output: data/evaluation/testbed_sample.csv
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "corpus" / "corpus.csv"
OUT_PATH = ROOT / "data" / "evaluation" / "testbed_sample.csv"

SAMPLE_SIZE = 1002
OUTLETS = ["GB News", "Daily Sceptic"]
SEED = 42

df = pd.read_csv(IN_PATH)

# Stratification variable: media outlet or think tank.
df["group"] = df["source"].apply(lambda s: "outlet" if s in OUTLETS else "Think tank")

sample, _ = train_test_split(df, train_size=SAMPLE_SIZE, stratify=df["group"], random_state=SEED)

sample.drop(columns=["group"]).to_csv(OUT_PATH, index=False)
