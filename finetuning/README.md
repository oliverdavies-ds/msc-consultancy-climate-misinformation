# 3. Fine tuning

Adapts the published CARDS classifier to UK discourse with a LoRA adapter.

| Order | Script | Input | Output |
|---|---|---|---|
| 1 | `generate_recot.py` | `data/finetuning/finetune_150.csv` (150 hand labelled statements) | `data/finetuning/uk_lora.jsonl` |
| 2 | `train_lora.py` | `uk_lora.jsonl` | `adapter/` |

The CARDS model is trained on examples that reason step by step before giving labels,
so new training examples need a reasoning trace in the same format. `generate_recot.py`
produces one for each statement and keeps it only if its final answer matches the gold
labels exactly.

`train_lora.py` uses the LoRA settings of the original CARDS training (rank 16, alpha
16, attention and MLP projections, 3 epochs, learning rate 2e-4), so the only thing
that changes is the data.

`adapter/adapter_config.json` records the trained configuration. The weights (about
320 MB) exceed the GitHub file size limit and are not in the repository.
