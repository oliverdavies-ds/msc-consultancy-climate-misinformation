"""
Prevalence and distribution of contrarian claims across the corpus (Figure 6
and the headline percentages in the results chapter).
"""

from itertools import chain
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

OUTLETS = ["GB News", "Daily Sceptic"]
CODE_PATTERN = r"\d+_\d+_\d+(?:_\d+)?"

df = pd.read_csv(ROOT / "data" / "public" / "corpus_results.csv")
df["not_claim"] = df["predicted_claims"].str.contains("0_0_0", na=False)

# Untouched copy for the individual claim plot at the end.
df1 = df.copy()

#%% Claim density

# Share of statements with and without a contrarian claim.
print(df["not_claim"].value_counts() / len(df))

# Pool the five think tanks into one group.
df["group"] = df["source"].where(df["source"].isin(OUTLETS), "Think tanks")

# Proportion of statements that carry a claim, by group and by individual source.
claim_dense_group = df.groupby("group")["not_claim"].apply(lambda x: (1 - x).mean())
claim_dense_source = df.groupby("source")["not_claim"].apply(lambda x: (1 - x).mean())
print(claim_dense_group, claim_dense_source, sep="\n")

#%% Super claim prevalence

# The super claim is the first level of a CARDS code (1 to 7).
df["super_codes"] = df["predicted_claims"].str.findall(CODE_PATTERN)
df["super_codes"] = df["super_codes"].apply(lambda codes: [c.split("_")[0] for c in codes])

# Number of claim bearing statements containing each super claim. Each super
# claim is counted once per statement.
sup_corp = pd.Series(chain.from_iterable(df.loc[~df["not_claim"], "super_codes"].apply(set))).value_counts()

print(sup_corp / len(df))                    # share of all statements
print(sup_corp / (~df["not_claim"]).sum())   # share of claim bearing statements

super_percentage_tot = sup_corp / len(df)
super_percentage_claim = sup_corp / sup_corp.sum()

# Science scepticism: super claims 1, 2, 3 and 5.
print(super_percentage_tot[["1", "2", "3", "5"]].sum())
print(super_percentage_claim[["1", "2", "3", "5"]].sum())

# Response scepticism: super claims 4 and 6.
print(super_percentage_tot[["4", "6"]].sum())
print(super_percentage_claim[["4", "6"]].sum())

#%% Super claim mix within each group and source

exploded = df.explode("super_codes")

superclaims_group = exploded.groupby(["group", "super_codes"]).size().unstack().drop(columns="0")
superclaims_group_pct = superclaims_group.div(superclaims_group.sum(axis=1), axis=0)

superclaims_source = exploded.groupby(["source", "super_codes"]).size().unstack().drop(columns="0")
superclaims_source_pct = superclaims_source.div(superclaims_source.sum(axis=1), axis=0)

print(superclaims_group_pct.round(3), superclaims_source_pct.round(3), sep="\n")

#%% Figure 6: the 20 most frequent individual claims

# One row per statement and claim pair.
df1["predicted_claims"] = df["predicted_claims"].str.split(",")
explod = df1.explode("predicted_claims")
explod["predicted_claims"] = explod["predicted_claims"].str.strip(" []'\"")
explod = explod[explod["predicted_claims"] != "0_0_0"]

top = explod["predicted_claims"].value_counts().head(20).sort_values()

# Bars are coloured by super claim.
super_to_colour = {
    "1": "orange",
    "2": "pink",
    "3": "orange",
    "4": "green",
    "5": "blue",
    "6": "red",
    "7": "purple",
}
bar_colours = [super_to_colour.get(s) for s in top.index.str.split("_").str[0]]

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top.index, top.values, color=bar_colours)
plt.xlabel("Number of Statements")
plt.ylabel("CARDS Taxonomies")
plt.title("Figure 6: 20 most frequent CARDS claims")
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.gca().set_axisbelow(True)
plt.tight_layout()
plt.savefig(FIGURES / "top20_claims.png")
plt.show()
