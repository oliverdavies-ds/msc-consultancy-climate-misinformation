# 1. Corpus construction

Turns raw source documents into one table of climate related statements.

| Order | Script | Input | Output |
|---|---|---|---|
| 1a | `scrape_daily_sceptic.py` | Daily Sceptic WordPress API | `data/interim/daily_sceptic_corpus.jsonl` |
| 1b | `clean_nexis_rtf.py` | `data/raw/gbnews_rtf/` | `data/interim/gbnews_articles_raw.jsonl` |
| 1b | `deduplicate_gbnews.py` | `gbnews_articles_raw.jsonl` | `data/interim/gbnews_corpus.jsonl` |
| 1c | `extract_pdf_text.py` | `data/raw/think_tank_pdfs/` | `data/interim/think_tanks_text.jsonl` |
| 1c | `map_pdf_metadata.py` | `think_tanks_text.jsonl`, `data/think_tank_manifest.xlsx` | `data/interim/think_tanks_corpus.jsonl` |
| 2 | `split_statements.py` | `{source}_corpus.jsonl` | `{source}_statements.jsonl` |
| 3 | `filter_climatebert.py` | `{source}_statements.jsonl` | `{source}_statements_climate.jsonl`, `{source}_statements_dropped.jsonl` |
| 4 | `merge_corpus.py` | `{source}_statements_climate.jsonl` | `data/corpus/corpus.csv` |
| 5 | `corpus_composition.py` | statements and corpus | Tables 1 and 2 |

`merge_corpus.py` finishes by running the data quality checks in `checks.py`, and stops
without saving if the corpus has missing values, duplicates, statements outside the 10 to
300 word range or dates outside the study period. The tests for these steps are in `tests/`.

Steps 1a to 1c are independent, one per source group. Every source ends in the same
schema (`doc_id`, `title`, `date_published`, `source`, `text`), so steps 2 to 4 treat
all three identically.

The final corpus has 44,883 statements: 27,384 from GB News, 9,712 from the Daily
Sceptic and 7,787 from the think tanks.
