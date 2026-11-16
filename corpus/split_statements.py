"""
Split each document into paragraph level statements, the unit of analysis
used by the CARDS classifier.

Input:  data/interim/{source}_corpus.jsonl
Output: data/interim/{source}_statements.jsonl
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"

SOURCES = ["gbnews", "daily_sceptic", "think_tanks"]
MIN_WORDS = 10
MAX_WORDS = 300

for source in SOURCES:
    in_path = INTERIM / f"{source}_corpus.jsonl"
    out_path = INTERIM / f"{source}_statements.jsonl"

    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            doc = json.loads(line)
            doc_id = doc.get("doc_id", "")

            # Split on line breaks. A block that starts in lower case is treated as a
            # paragraph broken across a page and is joined back onto the previous block.
            paragraphs = []
            for block in re.split(r"\n+", doc.get("text", "")):
                block = " ".join(block.split())
                if not block:
                    continue
                if paragraphs and block[:1].islower():
                    paragraphs[-1] += " " + block
                else:
                    paragraphs.append(block)

            i = 0
            for paragraph in paragraphs:
                # Fully upper case blocks are headings.
                if paragraph.isupper():
                    continue
                # Very short blocks carry no claim and very long ones are extraction errors.
                if not MIN_WORDS <= len(paragraph.split()) <= MAX_WORDS:
                    continue

                fout.write(json.dumps({
                    "statement_id": f"{doc_id}_{i}",
                    "source": doc.get("source", ""),
                    "date_published": doc.get("date_published") or doc.get("date", ""),
                    "title": doc.get("title", ""),
                    "statement": paragraph,
                }, ensure_ascii=False) + "\n")
                i += 1
