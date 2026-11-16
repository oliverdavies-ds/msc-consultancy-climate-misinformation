"""
Collect Daily Sceptic climate articles through the public WordPress REST API.

Output: data/interim/daily_sceptic_corpus.jsonl, one article per line.

Note: older articles have since moved behind a paywall, so a rerun today
returns fewer articles than the original collection.
"""

import json
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "interim" / "daily_sceptic_corpus.jsonl"

API_URL = "https://dailysceptic.org/wp-json/wp/v2/posts"
CLIMATE_CATEGORY = 3246              # WordPress category ID for climate change
DATE_FROM = "2023-09-01T00:00:00"    # window matched to the GB News collection
DATE_TO = "2026-06-30T00:00:00"

records = []
page = 1

# The API returns 100 posts per page, so keep requesting until a page comes back empty.
while True:
    posts = requests.get(API_URL, params={
        "categories": CLIMATE_CATEGORY,
        "per_page": 100,
        "page": page,
        "after": DATE_FROM,
        "before": DATE_TO,
        "orderby": "date",
        "order": "asc",
    }).json()

    # Past the last page the API returns an error object instead of a list.
    if not isinstance(posts, list) or not posts:
        break

    for post in posts:
        body = BeautifulSoup(post["content"]["rendered"], "html.parser")
        records.append({
            "doc_id": f"DS{len(records) + 1}",
            "title": BeautifulSoup(post["title"]["rendered"], "html.parser").get_text(),
            "date_published": post["date"][:10],
            "source": "Daily Sceptic",
            # Keep paragraph text only, which drops embeds, captions and navigation.
            "text": "\n\n".join(p.get_text(" ", strip=True) for p in body.find_all("p")),
        })

    print(f"page {page}: {len(records)} articles collected")
    page += 1

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
