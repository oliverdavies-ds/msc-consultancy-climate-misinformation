"""
Distribution of FLICC fallacies and their relationship to CARDS super claims
(Figures 14 and 15).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

df = pd.read_csv(ROOT / "data" / "public" / "statements_flicc.csv")

# Frequency of each fallacy, as counts and as a share of all claim bearing statements.
counts = df["flicc"].value_counts()
share = counts / counts.sum()

# Super claims present in each statement. A statement with several claims is
# counted once under each of them.
df["super_codes"] = df["predicted_claims"].str.findall(r"\d+_\d+_\d+(?:_\d+)?")
df["super_codes"] = df["super_codes"].apply(lambda codes: [c.split("_")[0] for c in codes])
df = df.explode("super_codes")

# Row percentages: the fallacy mix within each super claim.
table = pd.crosstab(df["super_codes"], df["flicc"])
row_pct = table.div(table.sum(axis=1), axis=0) * 100

plt.figure(figsize=(10, 5))
sns.heatmap(row_pct, annot=True, cmap="Greens")
plt.title("Figure 15: Heatmap of the percentage of FLICC claims by CARDS super-claim")
plt.xlabel("FLICC Fallacy Classification")
plt.ylabel("CARDS Super-Claim")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(FIGURES / "FLICC_HeatMap.png")
plt.show()
plt.close()

counts.sort_values().plot(kind="barh", figsize=(10, 5))
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.gca().set_axisbelow(True)
plt.title("Figure 14: Barchart of FLICC statement frequency")
plt.xlabel("Number of Statements")
plt.ylabel("FLICC Fallacy Classification")
plt.tight_layout()
plt.savefig(FIGURES / "FLICC_BAR.png")
plt.show()
plt.close()

# Combined share of the four most common fallacies.
print(share.sort_values(ascending=False).head(4).sum())
