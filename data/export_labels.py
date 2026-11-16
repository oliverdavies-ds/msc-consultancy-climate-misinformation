"""
Write the public, label only copies of the project outputs to data/public/.

The full outputs contain the text of news articles and think tank reports,
which is not redistributed in this repository. The copies written here keep
statement IDs, sources, dates and model outputs, which is everything the
analysis and evaluation scripts need.
"""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent
PUBLIC = DATA / "public"
PUBLIC.mkdir(exist_ok=True)

ID_COLUMNS = ["statement_id", "source", "date_published"]

# output name: (full file, columns to publish)
EXPORTS = {
    "corpus_results.csv": ("corpus/corpus_results.csv", ID_COLUMNS + ["predicted_claims"]),
    "statements_flicc.csv": ("corpus/statements_flicc.csv", ID_COLUMNS + ["predicted_claims", "flicc", "flicc_score"]),
    "statements_sentiment.csv": ("corpus/statements_sentiment.csv", ID_COLUMNS + ["predicted_claims", "sentiment", "sent_signed"]),
    "testbed_adjudication.csv": ("evaluation/testbed_adjudication.csv", ["statement_id", "source", "round_1", "round_2", "gold"]),
    "testbed_1002_base.csv": ("evaluation/testbed_1002_base.csv", ["statement_id", "source", "gold_claims", "predicted_claims"]),
    "testbed_1002_finetuned.csv": ("evaluation/testbed_1002_finetuned.csv", ["statement_id", "source", "gold_claims", "predicted_claims"]),
    "testbed_1002_zeroshot.csv": ("evaluation/testbed_1002_zeroshot.csv", ["statement_id", "source", "gold_claims", "predicted_claims"]),
    "congress_base.csv": ("evaluation/congress_base.csv", ["id", "claims", "predicted_claims"]),
    "congress_finetuned.csv": ("evaluation/congress_finetuned.csv", ["id", "claims", "predicted_claims"]),
    "debunkings.csv": ("debunking/debunk_code_samples.csv", ["code", "statement_id", "source", "date_published", "predicted_claims", "flicc", "debunking"]),
}

for name, (full_file, columns) in EXPORTS.items():
    pd.read_csv(DATA / full_file)[columns].to_csv(PUBLIC / name, index=False)
    print("wrote", name)
