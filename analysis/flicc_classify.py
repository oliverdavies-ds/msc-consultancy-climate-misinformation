"""
Label every claim bearing statement with a FLICC reasoning fallacy, using the
classifier released by Zanartu et al. (fzanartu/flicc).

Input:  data/corpus/corpus_claimbearing.csv
Output: data/corpus/statements_flicc.csv
"""

from pathlib import Path

import pandas as pd
from transformers import pipeline

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"

# Same settings as the ClimateBERT filter: truncate at 512 tokens and run on CPU.
classifier = pipeline("text-classification", model="fzanartu/flicc", truncation=True,
                      max_length=512, batch_size=32, device=-1)

df = pd.read_csv(CORPUS / "corpus_claimbearing.csv")

results = classifier(df["statement"].tolist())

df["flicc"] = [r["label"] for r in results]
df["flicc_score"] = [round(r["score"], 4) for r in results]

df.to_csv(CORPUS / "statements_flicc.csv", index=False)

print(df["flicc"].value_counts())
