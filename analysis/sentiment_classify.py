"""
Score the sentiment of every statement in the corpus with the Cardiff NLP
RoBERTa sentiment model.

Input:  data/corpus/corpus_results.csv
Output: data/corpus/statements_sentiment.csv
"""

from pathlib import Path

import pandas as pd
from transformers import pipeline

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"

classifier = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest", batch_size=30)

df = pd.read_csv(CORPUS / "corpus_results.csv")

results = classifier(df["statement"].tolist())

SIGN = {"positive": 1, "negative": -1, "neutral": 0}

df["sentiment"] = [r["label"] for r in results]

# Signed polarity: the label direction multiplied by the model confidence,
# giving a score between -1 (confidently negative) and +1 (confidently positive).
df["sent_signed"] = [SIGN[r["label"]] * r["score"] for r in results]

df.to_csv(CORPUS / "statements_sentiment.csv", index=False)

print(df["sentiment"].value_counts())
