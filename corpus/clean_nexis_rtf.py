"""
Parse GB News articles out of Nexis RTF exports.

Each export holds up to 100 articles wrapped in Nexis metadata. This script
strips the wrapper and keeps the title, publication date and body text.

Input:  data/raw/gbnews_rtf/*.RTF
Output: data/interim/gbnews_articles_raw.jsonl
"""

import json
import re
from datetime import datetime
from pathlib import Path

from striprtf.striprtf import rtf_to_text

ROOT = Path(__file__).resolve().parents[1]
RTF_DIR = ROOT / "data" / "raw" / "gbnews_rtf"
OUT_PATH = ROOT / "data" / "interim" / "gbnews_articles_raw.jsonl"

SOURCE = "GB News"

# Website navigation labels that Nexis carries into the body text.
NAVIGATION_LABELS = {"trending", "stories", "videos", "your say", "latest developments"}

# Nexis footer fields. The article body ends at the first of these.
FOOTER_PREFIXES = ("Load-Date:", "Our Standards:", "Classification", "Notes", "Graphic", "Highlight:")


def parse_article(block):
    """Return the date, title and body text of one Nexis article, or None if it is not a full article."""
    lines = [line.strip() for line in block.split("\n")]
    if SOURCE not in lines or "Body" not in lines:
        return None

    # Nexis always prints the headline on the last non empty line before the outlet name.
    title = next((line for line in reversed(lines[:lines.index(SOURCE)]) if line), "")

    # The first "Month day, year" string in the block is the publication date.
    date_match = re.search(r"([A-Z][a-z]+ \d{1,2}, \d{4})", block)
    date = datetime.strptime(date_match.group(1), "%B %d, %Y").strftime("%Y-%m-%d")

    paragraphs = []
    for line in lines[lines.index("Body") + 1:]:
        if line.startswith(FOOTER_PREFIXES):
            break
        # Skip blank lines, bullet lists and navigation labels.
        if not line or line.startswith(("•", "·", "*")) or line.lower() in NAVIGATION_LABELS:
            continue
        paragraphs.append(line)

    text = "\n\n".join(paragraphs)
    return {"date": date, "title": title, "text": text} if text else None


articles = []
for path in sorted(RTF_DIR.glob("*")):
    if path.suffix.lower() not in (".rtf", ".txt"):
        continue
    # Nexis closes every article with the marker "End of Document".
    for block in rtf_to_text(path.read_text(encoding="utf-8")).split("End of Document"):
        article = parse_article(block)
        if article:
            articles.append(article)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for article in articles:
        f.write(json.dumps(article, ensure_ascii=False) + "\n")
