# GPU runs

Classification, fine tuning and debunking were run on a rented GPU (RunPod).
Everything else runs on a laptop. The commands below assume the repository has
been cloned to `/workspace`, the `data/` folder has been uploaded into it, and
commands are run from the repository root.

```bash
export HF_HOME=/workspace/hf     # keep model downloads on the persistent volume
export HF_TOKEN=...              # your own Hugging Face token
```

## 1. Base CARDS model

Serves the published classifier on its own, for the baseline runs.

```bash
pip install vllm openai huggingface_hub

vllm serve C3DS/CARDS-Qwen3.6-27B --language-model-only \
    --max-model-len 8000 --max-num-seqs 50 --enable-prefix-caching
```

`--language-model-only` skips the vision tower, which is not needed for text.
`--max-model-len` and `--max-num-seqs` are set so that 50 concurrent requests
fit in GPU memory. Then, in a second terminal:

```bash
python classification/classify_cards.py --model C3DS/CARDS-Qwen3.6-27B \
    --input data/evaluation/testbed_1002.csv --output data/evaluation/testbed_1002_base.csv
```

## 2. Fine tuning the UK adapter

```bash
pip install unsloth trl datasets
python finetuning/train_lora.py
```

The adapter is written to `finetuning/adapter/`.

## 3. Fine tuned model

Serves the base model with the UK adapter attached under the name `cards-boost`.

```bash
vllm serve C3DS/CARDS-Qwen3.6-27B --language-model-only \
    --enable-lora --lora-modules cards-boost=finetuning/adapter --max-lora-rank 16 \
    --max-model-len 8000 --max-num-seqs 50 --enable-prefix-caching
```

```bash
# UK testbed
python classification/classify_cards.py \
    --input data/evaluation/testbed_1002.csv --output data/evaluation/testbed_1002_finetuned.csv

# full corpus (the default input and output)
python classification/classify_cards.py
```

## 4. Debunker

The debunker relies on the LangChain agent interface that was removed in
LangChain 1.0, so older versions are pinned.

```bash
pip install vllm 'langchain<1.0' 'langchain-openai<0.3' 'langchain-community<0.4' \
    sentence-transformers pandas tqdm 'ddgs==9.0.0' hf_transfer

vllm serve mistralai/Mistral-Small-24B-Instruct-2501 --served-model-name mistral-small
```

```bash
python debunking/run_debunker.py
```

## API runs

The zero shot baseline and the reasoning trace generator call the Anthropic API
and need no GPU.

```bash
export ANTHROPIC_API_KEY=...
pip install anthropic huggingface_hub

python classification/classify_zero_shot.py
python finetuning/generate_recot.py
```
