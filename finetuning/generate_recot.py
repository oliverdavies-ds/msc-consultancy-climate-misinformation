"""
Generate reasoning traces for the UK fine tuning set.

The CARDS model is trained on examples that reason step by step before giving
their labels. To fine tune it further, each of the 150 hand labelled UK
statements needs a trace in the same format. Claude Sonnet 4.6 is given the
statement and its gold labels and asked to write the reasoning as though
reaching those labels unaided. A trace is kept only if its final answer
contains exactly the gold labels.

Requires the ANTHROPIC_API_KEY environment variable.

Input:  data/finetuning/finetune_150.csv
Output: data/finetuning/uk_lora.jsonl, in chat format ready for training
"""

import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from anthropic import Anthropic
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "finetuning" / "finetune_150.csv"
OUT_PATH = ROOT / "data" / "finetuning" / "uk_lora.jsonl"

MODEL = "claude-sonnet-4-6"
BASE_MODEL = "C3DS/CARDS-Qwen3.6-27B"
GOLD_COLUMN = "gold_claims"
MAX_TOKENS = 4000    # same budget as the CARDS paper
WORKERS = 8
ATTEMPTS = 5

CODE = re.compile(r"\d+_\d+_\d+(?:_\d+)?")

# System instruction and chain of thought trigger released with the CARDS model.
prompts = json.load(open(hf_hub_download(BASE_MODEL, "cards_prompts.json")))
SYSTEM_INSTRUCTION = prompts["slim_system_instruction"]
COT_TRIGGER = prompts["cot_trigger"]

client = Anthropic()   # reads ANTHROPIC_API_KEY from the environment

rows = list(csv.DictReader(open(IN_PATH, encoding="utf-8")))

# (statement, gold codes) pairs. The regex handles single and multiple label cells alike.
data = [(r["statement"].strip(), list(dict.fromkeys(CODE.findall(str(r[GOLD_COLUMN])))))
        for r in rows if r.get("statement", "").strip()]


def generate_trace(item):
    """Return a reasoning trace ending in the gold labels, or None if no attempt is valid."""
    text, gold = item
    user = (f"### Text:\n{text}\n\n### True Labels:\n{', '.join(gold)}\n\n{COT_TRIGGER}\n\n"
            f"[You are writing the reasoning  for a training example. Write the <think>  reasoning "
            f"(CLAIMS / CONTEXT / SCAN / VERIFY) as if you are analysing the text for the first time and independently arriving at the true labels above — do NOT indicate that you were given the labels. Then output the YAML containing exactly those categories and no others.]")

    for _ in range(ATTEMPTS):
        try:
            reply = client.messages.create(model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM_INSTRUCTION,
                                           messages=[{"role": "user", "content": user}]).content[0].text

            # Discard anything written before the opening <think> tag.
            if "<think>" in reply:
                reply = reply[reply.index("<think>"):]

            # Accept the trace only if the answer after the reasoning matches the gold labels exactly.
            answer = reply.split("</think>")[-1]
            if set(CODE.findall(answer)) == set(gold):
                return reply
        except Exception:
            time.sleep(2)
    return None


with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    traces = list(executor.map(generate_trace, data))

# One training example per line: system prompt, the user turn exactly as it
# appears at inference, and the generated trace as the assistant turn.
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for (text, gold), trace in zip(data, traces):
        if trace is None:
            continue
        f.write(json.dumps({"messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": f"### Text:\n{text}\n\n{COT_TRIGGER}"},
            {"role": "assistant", "content": trace},
        ]}, ensure_ascii=False) + "\n")
