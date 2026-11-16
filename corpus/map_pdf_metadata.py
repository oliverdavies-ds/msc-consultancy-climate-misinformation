"""
Attach publication metadata to the extracted think tank texts.

The manifest is a hand built record of every PDF with its think tank,
publication date and title.

Input:  data/interim/think_tanks_text.jsonl, data/think_tank_manifest.xlsx
Output: data/interim/think_tanks_corpus.jsonl
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "think_tank_manifest.xlsx"
IN_PATH = ROOT / "data" / "interim" / "think_tanks_text.jsonl"
OUT_PATH = ROOT / "data" / "interim" / "think_tanks_corpus.jsonl"

manifest = pd.read_excel(MANIFEST_PATH)

# Dates are recorded day first in the manifest. Convert to ISO format.
manifest["date_published"] = pd.to_datetime(manifest["date"], dayfirst=True).dt.strftime("%Y-%m-%d")

# Look up table from file name to metadata.
metadata = manifest.set_index("file")[["source", "date_published", "title"]].to_dict("index")

records = []
for line in open(IN_PATH, encoding="utf-8"):
    doc = json.loads(line)
    meta = metadata.get(doc["file"], {})
    records.append({
        "doc_id": f"TT{len(records) + 1}",
        "title": meta.get("title", ""),
        "date_published": meta.get("date_published", ""),
        "source": meta.get("source", ""),
        "file": doc["file"],
        "text": doc["text"],
    })

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
