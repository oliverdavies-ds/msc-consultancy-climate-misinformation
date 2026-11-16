"""
Zero shot baseline: classify the UK testbed with Claude Sonnet 4.6 through the
Anthropic API, using the same CARDS prompt as the fine tuned models.

Requires the ANTHROPIC_API_KEY environment variable.

Input:  data/evaluation/testbed_1002.csv
Output: data/evaluation/testbed_1002_zeroshot.csv
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
IN_PATH = ROOT / "data" / "evaluation" / "testbed_1002.csv"
OUT_PATH = ROOT / "data" / "evaluation" / "testbed_1002_zeroshot.csv"

MODEL = "claude-sonnet-4-6"
PROMPT_SOURCE = "C3DS/CARDS-Qwen3.6-27B"
MAX_TOKENS = 4000
WORKERS = 10        # kept low to stay inside the API rate limit
ATTEMPTS = 3

CODE = re.compile(r"\d+_\d+_\d+(?:_\d+)?")

# Same system instruction and chain of thought trigger as the CARDS model.
prompts = json.load(open(hf_hub_download(PROMPT_SOURCE, "cards_prompts.json")))
SYSTEM_INSTRUCTION = prompts["slim_system_instruction"]
COT_TRIGGER = prompts["cot_trigger"]

client = Anthropic()   # reads ANTHROPIC_API_KEY from the environment

rows = list(csv.DictReader(open(IN_PATH, encoding="utf-8")))


def classify(row):
    """Return the list of CARDS codes predicted for one statement."""
    for _ in range(ATTEMPTS):
        try:
            # The long system instruction is identical on every call, so it is cached to cut cost.
            reply = client.messages.create(
                model=MODEL, max_tokens=MAX_TOKENS, temperature=0,
                system=[{"type": "text", "text": SYSTEM_INSTRUCTION, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": f"### Text:\n{row['statement']}\n\n{COT_TRIGGER}"}],
            ).content[0].text
            # Read codes from the final answer only, after the reasoning trace.
            answer = reply.split("</think>")[-1] if "</think>" in reply else reply
            return list(dict.fromkeys(CODE.findall(answer))) or ["0_0_0"]
        except Exception:
            print("Request failed, retrying", flush=True)
            time.sleep(3)

    return ["ERROR"]


with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    predictions = list(executor.map(classify, rows))

with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["predicted_claims"])
    writer.writeheader()
    for row, codes in zip(rows, predictions):
        writer.writerow({**row, "predicted_claims": "['" + "', '".join(codes) + "']"})
