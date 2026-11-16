"""
Keep only climate related statements, using the ClimateBERT climate detector.

Every statement is scored with the probability that it is about climate.
Statements at or above the threshold are kept and the rest are written to a
separate file so the filter can be audited.

Input:  data/interim/{source}_statements.jsonl
Output: data/interim/{source}_statements_climate.jsonl
        data/interim/{source}_statements_dropped.jsonl
"""

import json
from pathlib import Path

from transformers import pipeline

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"

SOURCES = ["gbnews", "daily_sceptic", "think_tanks"]
MODEL = "climatebert/distilroberta-base-climate-detector"
THRESHOLD = 0.5

# Inputs are truncated at the model limit of 512 tokens. device=-1 runs on CPU,
# which was fast enough for this corpus on a laptop.
detector = pipeline("text-classification", model=MODEL, truncation=True,
                    max_length=512, batch_size=30, device=-1)

for source in SOURCES:
    in_path = INTERIM / f"{source}_statements.jsonl"
    keep_path = INTERIM / f"{source}_statements_climate.jsonl"
    drop_path = INTERIM / f"{source}_statements_dropped.jsonl"

    statements = [json.loads(line) for line in open(in_path, encoding="utf-8") if line.strip()]
    texts = [s["statement"] for s in statements]

    with open(keep_path, "w", encoding="utf-8") as keep, open(drop_path, "w", encoding="utf-8") as drop:
        for statement, result in zip(statements, detector(texts)):
            # The model returns the winning label with its score. Convert to P(climate).
            prob = result["score"] if result["label"].lower() == "yes" else 1 - result["score"]
            statement["climate_prob"] = round(prob, 4)

            target = keep if prob >= THRESHOLD else drop
            target.write(json.dumps(statement, ensure_ascii=False) + "\n")
