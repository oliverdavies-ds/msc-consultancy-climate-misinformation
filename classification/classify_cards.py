"""
Classify statements against the CARDS taxonomy with a model served by vLLM.

One script covers every classification run in the project. The served model
and the input file are chosen on the command line:

    base model on the UK testbed
        python classification/classify_cards.py --model C3DS/CARDS-Qwen3.6-27B \
            --input data/evaluation/testbed_1002.csv --output data/evaluation/testbed_1002_base.csv

    fine tuned adapter on the UK testbed
        python classification/classify_cards.py \
            --input data/evaluation/testbed_1002.csv --output data/evaluation/testbed_1002_finetuned.csv

    fine tuned adapter on the full corpus (the defaults)
        python classification/classify_cards.py

For the CARDS congressional test set add --text-column text or statement as
needed. See docs/gpu_runs.md for the vLLM commands that serve each model.
"""

import argparse
import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import hf_hub_download
from openai import OpenAI

BASE_MODEL = "C3DS/CARDS-Qwen3.6-27B"   # published CARDS classifier and source of the prompts
ADAPTER_NAME = "cards-boost"            # name the UK LoRA adapter is served under
VLLM_URL = "http://localhost:8000/v1"
MAX_TOKENS = 4000                       # same budget as the CARDS paper, leaves room for the reasoning trace
WORKERS = 50                            # concurrent requests, matched to --max-num-seqs on the server
ATTEMPTS = 3

# A CARDS code has three levels, with an optional fourth (for example 4_1_2 or 4_2_1_1).
CODE = re.compile(r"\d+_\d+_\d+(?:_\d+)?")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--input", default="data/corpus/corpus.csv")
parser.add_argument("--output", default="data/corpus/corpus_results.csv")
parser.add_argument("--model", default=ADAPTER_NAME, help="model name as served by vLLM")
parser.add_argument("--text-column", default="statement")
args = parser.parse_args()

# The system instruction and chain of thought trigger are the ones released with
# the CARDS model, so every model in the comparison receives an identical prompt.
prompts = json.load(open(hf_hub_download(BASE_MODEL, "cards_prompts.json")))
SYSTEM_INSTRUCTION = prompts["slim_system_instruction"]
COT_TRIGGER = prompts["cot_trigger"]

# vLLM exposes an OpenAI compatible endpoint. It does not check the key, but the client requires one.
client = OpenAI(base_url=VLLM_URL, api_key="not-needed")

rows = list(csv.DictReader(open(args.input, encoding="utf-8")))


def classify(row):
    """Return the list of CARDS codes predicted for one statement."""
    for _ in range(ATTEMPTS):
        try:
            # Temperature 0 keeps the output deterministic and reproducible.
            reply = client.chat.completions.create(
                model=args.model, temperature=0, max_tokens=MAX_TOKENS,
                messages=[{"role": "system", "content": SYSTEM_INSTRUCTION},
                          {"role": "user", "content": f"### Text:\n{row[args.text_column]}\n\n{COT_TRIGGER}"}],
            ).choices[0].message.content
        except Exception as exc:
            print("Request failed:", exc, flush=True)
            time.sleep(3)
            continue

        # The model reasons inside <think> tags and then gives its answer. Codes are
        # read from the answer only, so codes it considered and rejected are not counted.
        answer = reply.split("</think>")[-1] if "</think>" in reply else reply

        # Deduplicate while keeping order. An answer with no code means no contrarian claim.
        return list(dict.fromkeys(CODE.findall(answer))) or ["0_0_0"]

    # Every attempt failed. Flag the row so it can be found and rerun.
    return ["ERROR"]


# executor.map returns results in input order, so predictions stay aligned with rows.
with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    predictions = list(executor.map(classify, rows))

with open(args.output, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["predicted_claims"])
    writer.writeheader()
    for row, codes in zip(rows, predictions):
        # Stored as a Python style list string, the format of the gold labels.
        writer.writerow({**row, "predicted_claims": "['" + "', '".join(codes) + "']"})

print("Rows that failed after all attempts:", sum(p == ["ERROR"] for p in predictions))
