"""
Mean sentiment of contrarian claims by super claim and source (Figure 8).
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

SUPER_CLAIM_NAMES = {"1": "1 Not happening", "2": "2 Not us", "3": "3 Not bad", "4": "4 Solutions bad",
                     "5": "5 Science unreliable", "6": "6 Actors biased", "7": "7 Fossil fuels good"}

df = pd.read_csv(ROOT / "data" / "public" / "statements_sentiment.csv")

# Pool the five think tanks into one group.
df["group"] = df["source"].where(df["source"].isin(["GB News", "Daily Sceptic"]), "Think tanks")

# Super claims present in each statement, excluding the no claim code.
df["super"] = df["predicted_claims"].apply(
    lambda s: sorted({c[0] for c in re.findall(r"\d+_\d+_\d+(?:_\d+)?", str(s))} - {"0"}))

df["has_claim"] = ~df["predicted_claims"].str.contains("0_0_0")

# Sanity check: claim bearing statements should be more negative on average.
print(df.groupby("has_claim")["sent_signed"].mean())

# One row per statement and super claim pair.
df = df.explode("super").dropna(subset=["super"])
df["super"] = df["super"].map(SUPER_CLAIM_NAMES)

pivot = df.pivot_table(index="super", columns="group", values="sent_signed", aggfunc="mean")
counts = df.pivot_table(index="super", columns="group", values="sent_signed", aggfunc="count")

pivot["All sources"] = df.groupby("super")["sent_signed"].mean()
counts["All sources"] = df.groupby("super")["sent_signed"].count()

plt.figure(figsize=(9, 5))
sns.heatmap(pivot, annot=True, cmap="Reds_r", cbar_kws={"label": "Mean sentiment (-1 negative, +1 positive)"})
plt.title("Figure 8: Heatmap of mean sentiment by super-claim and source")
plt.xlabel("Misinformation Source")
plt.ylabel("CARDS Super Taxonomy")
plt.tight_layout()
plt.savefig(FIGURES / "sentiment_source.png")
plt.show()
