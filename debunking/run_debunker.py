"""
Generate a debunking for each sampled statement.

Run on the GPU machine with the Mistral model served by vLLM. See docs/gpu_runs.md.

Input:  data/debunking/code_samples.csv (statements with CARDS and FLICC labels)
Output: data/debunking/debunk_code_samples.csv
"""

import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from core import HamburgerStyle

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "debunking" / "code_samples.csv"
OUT_PATH = ROOT / "data" / "debunking" / "debunk_code_samples.csv"

debunker = HamburgerStyle()

df = pd.read_csv(IN_PATH)

debunkings = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    # The first predicted CARDS code selects the supporting evidence.
    codes = re.findall(r"\d+_\d+_\d+(?:_\d+)?", str(row.predicted_claims))
    debunkings.append(debunker.rebuttal_generator(row.statement, row.flicc, codes[0]))

df["debunking"] = debunkings

df.to_csv(OUT_PATH, index=False)
