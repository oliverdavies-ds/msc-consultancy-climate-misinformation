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


def split_paragraphs(text):
    """Split text on line breaks and tidy the whitespace in each block."""
    paragraphs = []
    for block in re.split(r"\n+", text):
        block = " ".join(block.split())
        if not block:
            continue
        # A block that starts in lower case is treated as a paragraph broken across a
        # page and is joined back onto the previous block.
        if paragraphs and block[:1].islower():
            paragraphs[-1] += " " + block
        else:
            paragraphs.append(block)
    return paragraphs


def keep_paragraph(paragraph):
    """Decide whether a paragraph is kept as a statement."""
    # Fully upper case blocks are headings.
    if paragraph.isupper():
        return False
    # Very short blocks carry no claim and very long ones are extraction errors.
    return MIN_WORDS <= len(paragraph.split()) <= MAX_WORDS


def split_document(doc):
    """Turn one document into a list of statements, numbered within the document."""
    doc_id = doc.get("doc_id", "")
    kept = [p for p in split_paragraphs(doc.get("text", "")) if keep_paragraph(p)]
    return [{
        "statement_id": f"{doc_id}_{i}",
        "source": doc.get("source", ""),
        "date_published": doc.get("date_published") or doc.get("date", ""),
        "title": doc.get("title", ""),
        "statement": paragraph,
    } for i, paragraph in enumerate(kept)]


if __name__ == "__main__":
    for source in SOURCES:
        in_path = INTERIM / f"{source}_corpus.jsonl"
        out_path = INTERIM / f"{source}_statements.jsonl"

        with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                for statement in split_document(json.loads(line)):
                    fout.write(json.dumps(statement, ensure_ascii=False) + "\n")
