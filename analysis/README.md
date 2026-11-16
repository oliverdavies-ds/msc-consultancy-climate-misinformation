# 6. Analysis

| Script | Question | Output |
|---|---|---|
| `filter_claim_bearing.py` | Which statements carry a claim? | `data/corpus/corpus_claimbearing.csv` |
| `claim_distribution.py` | How common is each claim and super claim? | Figure 6 |
| `source_composition.R` | Does the claim mix differ by source? (chi squared, Cramer's V) | Figure 7 |
| `sentiment_classify.py` | Sentiment of every statement | `data/corpus/statements_sentiment.csv` |
| `sentiment_analysis.py` | How negative is each super claim, by source? | Figure 8 |
| `temporal_analysis.R` | How do claims change over time? (trends, volatility, January 2026 shift, bigrams) | Figures 9 to 13, A1, A2 and Tables 6 to 10 |
| `flicc_classify.py` | Reasoning fallacy of every claim bearing statement | `data/corpus/statements_flicc.csv` |
| `flicc_analysis.py` | Which fallacies accompany which claims? | Figures 14 and 15 |

The two `_classify` scripts and `filter_claim_bearing.py` need statement text and so
need the full local corpus. Every other script runs from `data/public/`. The bigram
section of `temporal_analysis.R` is skipped when statement text is not available.

Python scripts can be run from anywhere. R scripts should be run from the repository
root. Figures are written to `figures/` and regression tables to `tables/`.
