"""
Remove duplicate GB News articles and assign document IDs.

The Nexis searches were run in three keyword batches, so the same article can
appear more than once. Duplicates are identified by identical body text.

Input:  data/interim/gbnews_articles_raw.jsonl
Output: data/interim/gbnews_corpus.jsonl
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "interim" / "gbnews_articles_raw.jsonl"
OUT_PATH = ROOT / "data" / "interim" / "gbnews_corpus.jsonl"

articles = [json.loads(line) for line in open(IN_PATH, encoding="utf-8")]

seen_texts = set()
kept = []

for article in articles:
    if article["text"] in seen_texts:
        continue
    seen_texts.add(article["text"])
    # Same schema as the Daily Sceptic and think tank corpora.
    article["doc_id"] = f"GB{len(kept) + 1}"
    article["source"] = "GB News"
    article["date_published"] = article.pop("date")
    kept.append(article)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for article in kept:
        f.write(json.dumps(article, ensure_ascii=False) + "\n")
