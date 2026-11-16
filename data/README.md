# Data

## What is in this repository

`public/` holds label only copies of every output the analysis and evaluation
scripts need. Each file keeps statement IDs, sources, dates and model outputs
and leaves out the statement text.

| File | Contents |
|---|---|
| `corpus_results.csv` | CARDS codes predicted by the UK fine tuned model for all 44,883 statements |
| `statements_flicc.csv` | FLICC fallacy label and score for the 13,977 claim bearing statements |
| `statements_sentiment.csv` | Sentiment label and signed score for every statement |
| `testbed_adjudication.csv` | Both hand labelling rounds and the adjudicated gold label for the 1,002 statement UK testbed |
| `testbed_1002_base.csv`, `_finetuned.csv`, `_zeroshot.csv` | Gold labels and predictions of each model on the UK testbed |
| `congress_base.csv`, `congress_finetuned.csv` | Gold labels and predictions on the CARDS congressional test set |
| `debunkings.csv` | Generated debunkings for one sampled statement per CARDS code |

`think_tank_manifest.xlsx` lists the 210 think tank publications in the corpus
with their source, publication date and title.

`export_labels.py` regenerates `public/` from the full local outputs.

## What is not included

The text of the corpus is not redistributed. GB News articles were obtained
through Nexis under an institutional licence, and the Daily Sceptic articles
and think tank reports remain the copyright of their publishers. The hand
labelled testbed and the fine tuning set contain that text, so they are held
locally as well.

The CARDS congressional test set and the zero shot results of Coan et al. are
available from the CARDS repository.

## Rebuilding the corpus

The scripts in `corpus/` rebuild everything from the original sources and
expect this layout:

```
data/
├── raw/
│   ├── gbnews_rtf/          Nexis RTF exports of GB News articles
│   └── think_tank_pdfs/     the PDFs listed in think_tank_manifest.xlsx
├── interim/                 parsed articles and statements (created by the scripts)
├── corpus/                  merged corpus and model outputs
├── evaluation/              testbed and model predictions with text
├── finetuning/              150 labelled UK statements and their reasoning traces
└── debunking/               sampled statements and their debunkings
```

Daily Sceptic articles are collected by `corpus/scrape_daily_sceptic.py`. Some
older articles have since moved behind a paywall, so a new collection will be
smaller than the original.
