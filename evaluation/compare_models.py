"""
Compare the base CARDS model, the UK fine tuned adapter and the zero shot
baseline (Tables 3 and 4 and Figures 4 and 5 of the dissertation).

Two test sets are used:
  1. the CARDS congressional test set, to check that fine tuning on UK data
     does not degrade performance on the original US benchmark
  2. the hand labelled UK testbed of 1,002 statements

Metrics follow the CARDS paper: sample averaged F1, precision and recall,
exact match accuracy, Hamming loss and the Matthews correlation coefficient.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import (accuracy_score, f1_score, hamming_loss,
                             matthews_corrcoef, precision_score, recall_score)
from sklearn.preprocessing import MultiLabelBinarizer

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "data" / "public"
FIGURES = ROOT / "figures"

CODE = re.compile(r"\d+_\d+_\d+(?:_\d+)?")


def parse(cell):
    """Set of CARDS codes in a cell, truncated to level 3. An empty cell means no claim."""
    return {"_".join(code.split("_")[:3]) for code in CODE.findall(str(cell))} or {"0_0_0"}


def score(y_true, y_pred, name):
    """Multi label metrics for one model, as a table row."""
    return {"model": name,
            "F1": f1_score(y_true, y_pred, average="samples"),
            "precision": precision_score(y_true, y_pred, average="samples"),
            "recall": recall_score(y_true, y_pred, average="samples"),
            "accuracy": accuracy_score(y_true, y_pred),
            "hamming": hamming_loss(y_true, y_pred),
            "MCC": matthews_corrcoef(y_true.ravel(), y_pred.ravel())}


#%% Congressional test set (Table 3)

base_cong = pd.read_csv(PUBLIC / "congress_base.csv").rename(columns={"predicted_claims": "base_pred"})
fine_cong = pd.read_csv(PUBLIC / "congress_finetuned.csv").rename(columns={"predicted_claims": "fine_pred"})
df_cong = fine_cong.merge(base_cong[["id", "base_pred"]], on="id")

# Convert each set of codes into a binary indicator vector so that standard
# multi label metrics can be applied.
mlb = MultiLabelBinarizer().fit([parse(x) for col in ["claims", "base_pred", "fine_pred"] for x in df_cong[col]])
yt = mlb.transform([parse(x) for x in df_cong.claims])

congress_results = pd.DataFrame(
    [score(yt, mlb.transform([parse(x) for x in df_cong[m + "_pred"]]), m) for m in ["base", "fine"]]
).set_index("model").round(3)
print("Congressional test set\n", congress_results, "\n")

#%% UK testbed (Table 4)

base_df = pd.read_csv(PUBLIC / "testbed_1002_base.csv").rename(columns={"predicted_claims": "base_pred"})
fine_df = pd.read_csv(PUBLIC / "testbed_1002_finetuned.csv").rename(columns={"predicted_claims": "fine_pred"})
zero_df = pd.read_csv(PUBLIC / "testbed_1002_zeroshot.csv").rename(columns={"predicted_claims": "zero_pred"})

df = fine_df.merge(base_df[["statement_id", "base_pred"]], on="statement_id")
df = df.merge(zero_df[["statement_id", "zero_pred"]], on="statement_id")

mlb = MultiLabelBinarizer().fit(
    [parse(x) for col in ["gold_claims", "base_pred", "fine_pred", "zero_pred"] for x in df[col]])
yt = mlb.transform([parse(x) for x in df.gold_claims])

uk_results = pd.DataFrame(
    [score(yt, mlb.transform([parse(x) for x in df[m + "_pred"]]), m) for m in ["base", "fine", "zero"]]
).set_index("model").round(3)
print("UK testbed\n", uk_results, "\n")


# Wilcoxon signed rank test on per statement F1, fine tuned against base.
def f1_per_row(yp):
    return 2 * (yt & yp).sum(1) / (yt.sum(1) + yp.sum(1))


f1_base = f1_per_row(mlb.transform([parse(v) for v in df.base_pred]))
f1_fine = f1_per_row(mlb.transform([parse(v) for v in df.fine_pred]))
print(wilcoxon(f1_fine, f1_base))

#%% Where the fine tuned model corrects the base model (Figures 4 and 5)

disagree = df[[not (parse(g) == parse(b) == parse(f) == parse(z))
               for g, b, f, z in zip(df.gold_claims, df.base_pred, df.fine_pred, df.zero_pred)]]
print("Statements where at least one model departs from gold:", len(disagree))

# For statements the fine tuned model labels exactly right, count the codes it
# added to, and removed from, the base model prediction.
added = {}
removed = {}

for i in range(len(df)):
    gold_claims = parse(df.gold_claims[i])
    base_claims = parse(df.base_pred[i])
    fine_claims = parse(df.fine_pred[i])

    if fine_claims != gold_claims:
        continue

    for claim in fine_claims - base_claims:
        added[claim] = added.get(claim, 0) + 1
    for claim in base_claims - fine_claims:
        removed[claim] = removed.get(claim, 0) + 1

top_added = sorted(added.items(), key=lambda x: x[1], reverse=True)[:10]
top_removed = sorted(removed.items(), key=lambda x: x[1], reverse=True)[:10]

plt.figure(figsize=(8, 7))
plt.bar([c for c, _ in top_added], [n for _, n in top_added], color="green")
plt.ylabel("Frequency")
plt.xlabel("CARDS Taxonomies")
plt.title("Figure 4: Claims added by the fine-tuned model to match gold claims")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.gca().set_axisbelow(True)
plt.savefig(FIGURES / "Claims_added.png")
plt.show()

plt.figure(figsize=(8, 7))
plt.bar([c for c, _ in top_removed], [n for _, n in top_removed], color="darkred")
plt.ylabel("Frequency")
plt.xlabel("CARDS Taxonomies")
plt.title("Figure 5: Claims removed by the fine-tuned model to match gold claims")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.gca().set_axisbelow(True)
plt.savefig(FIGURES / "Claims_removed.png")
plt.show()

#%% Zero shot behaviour on the congressional set, claim against no claim rows

# Uses the zero shot results published with the CARDS paper (Coan et al.), which
# are not redistributed here. Download zeroshot_results.csv from the CARDS
# repository and save it as data/evaluation/congress_zeroshot_coan.csv to run this cell.
ZERO_PATH = ROOT / "data" / "evaluation" / "congress_zeroshot_coan.csv"

if ZERO_PATH.exists():
    zero_cong = pd.read_csv(ZERO_PATH)
    zero_cong = zero_cong[zero_cong["model"] == "Claude-3.7"]
    zero_cong = zero_cong.rename(columns={"predicted_claims": "zero_pred"})
    zero_cong = zero_cong.merge(df_cong[["id", "claims"]], on="id")

    claims_list = [parse(v) for v in zero_cong["claims"]]
    zero_pred_list = [parse(v) for v in zero_cong["zero_pred"]]

    mlb = MultiLabelBinarizer().fit(claims_list + zero_pred_list)
    y_true = mlb.transform(claims_list)
    y_pred_zero = mlb.transform(zero_pred_list)

    # Split the test set into statements with and without a contrarian claim.
    no_claim_mask = np.array([claim_set == {"0_0_0"} for claim_set in claims_list])
    has_claim_mask = ~no_claim_mask

    pct_noclaim = 100 * no_claim_mask.mean()
    acc_claim_zero = accuracy_score(y_true[has_claim_mask], y_pred_zero[has_claim_mask])
    acc_noclaim_zero = accuracy_score(y_true[no_claim_mask], y_pred_zero[no_claim_mask])

    print("Share of congressional statements with no claim (%):", round(pct_noclaim, 1))
    print("Zero shot exact match accuracy on claim rows:", round(acc_claim_zero, 3))
    print("Zero shot exact match accuracy on no claim rows:", round(acc_noclaim_zero, 3))
