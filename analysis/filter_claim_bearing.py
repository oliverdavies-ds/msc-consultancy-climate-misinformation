"""
Keep only the statements that carry at least one contrarian claim.

These are the statements passed to the FLICC fallacy classifier.

Input:  data/corpus/corpus_results.csv
Output: data/corpus/corpus_claimbearing.csv
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"

df = pd.read_csv(CORPUS / "corpus_results.csv")

# 0_0_0 is the CARDS code for "no contrarian claim".
df = df[~df["predicted_claims"].str.contains("0_0_0", na=False)]

df.to_csv(CORPUS / "corpus_claimbearing.csv", index=False)
