"""
Extract and clean body text from the think tank PDF publications.

Input:  data/raw/think_tank_pdfs/*.pdf
Output: data/interim/think_tanks_text.jsonl, one publication per line
"""

import json
import os
import re
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "raw" / "think_tank_pdfs"
OUT_PATH = ROOT / "data" / "interim" / "think_tanks_text.jsonl"

# Paragraphs containing publishing boilerplate (copyright notices, ISBNs, disclaimers).
BOILERPLATE = re.compile(r"©|copyright|rights reserved|\bISBN\b|\bISSN\b|views expressed|registered charity", re.I)

# Numbered footnotes that are only a reference to a web source.
REFERENCE = re.compile(r"^\d{1,3}\s.*(https?://|www\.|Accessed )")

records = []
for name in sorted(os.listdir(PDF_DIR)):
    if not name.lower().endswith(".pdf"):
        continue

    doc = fitz.open(os.path.join(PDF_DIR, name))
    text = "\n".join(page.get_text() for page in doc)
    doc.close()

    # PDF text arrives as hard wrapped lines, so paragraphs have to be rebuilt.
    text = re.sub(r"-\n(\w)", r"\1", text)               # rejoin words hyphenated across lines
    text = re.sub(r"(?<=[.!?:])\s*\n", "\n\n", text)     # a line ending in punctuation closes a paragraph
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)         # any other single line break is a soft wrap
    text = re.sub(r"\n{3,}", "\n\n", text)               # collapse runs of blank lines

    paragraphs = []
    for paragraph in text.split("\n\n"):
        paragraph = " ".join(paragraph.split())
        if BOILERPLATE.search(paragraph) or REFERENCE.match(paragraph):
            continue
        # Drop fragments such as page numbers, table cells and chart labels,
        # identified as having fewer than five words of three or more letters.
        if len(re.findall(r"[a-zA-Z]{3,}", paragraph)) < 5:
            continue
        paragraphs.append(paragraph)

    records.append({"file": name, "text": "\n\n".join(paragraphs)})

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
