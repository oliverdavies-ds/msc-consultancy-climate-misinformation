# 2 and 5. Testbed and model comparison

| Script | Purpose | Reads |
|---|---|---|
| `sample_testbed.py` | Draws the stratified sample for hand labelling | `data/corpus/corpus.csv` |
| `intra_rater_kappa.py` | Cohen's kappa between the two labelling rounds | `data/public/testbed_adjudication.csv` |
| `compare_models.py` | Metrics, significance test and error analysis for all models | `data/public/testbed_1002_*.csv`, `data/public/congress_*.csv` |

The testbed has 1,002 statements, 334 from each of GB News, the Daily Sceptic and the
think tanks. It was labelled twice, four weeks apart, and a final gold label was then
settled for each statement.

`compare_models.py` reports results on two test sets: the UK testbed, and the CARDS
congressional test set as a check on what fine tuning costs on the original benchmark.
It also plots which codes the adapter adds and removes to match the gold labels
(Figures 4 and 5).

`intra_rater_kappa.py` and `compare_models.py` run from the public data as provided.
