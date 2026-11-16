# 4. Classification

| Script | Model | Runs on |
|---|---|---|
| `classify_cards.py` | Published CARDS model, or CARDS with the UK adapter | GPU, served with vLLM |
| `classify_zero_shot.py` | Claude Sonnet 4.6 | Anthropic API |

`classify_cards.py` is used for every run in the project. The model, input file and
output file are command line arguments, and the defaults classify the full corpus with
the UK adapter:

```bash
python classification/classify_cards.py --model C3DS/CARDS-Qwen3.6-27B \
    --input data/evaluation/testbed_1002.csv --output data/evaluation/testbed_1002_base.csv
```

All models receive the same system instruction and reasoning trigger, downloaded from
the CARDS model repository, at temperature 0. Each reasons before answering, and codes
are read from the final answer only. Output is the input file with one added column,
`predicted_claims`.

Serving commands are in [`docs/gpu_runs.md`](../docs/gpu_runs.md).
