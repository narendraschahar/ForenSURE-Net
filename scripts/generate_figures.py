import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


Path("figures").mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# Figure 1: Reliability Diagram
test = load_json("results/residual_stegnet_test_results.json")
bins = test["reliability_bins"]

conf = [b["confidence"] for b in bins if b["confidence"] is not None]
acc = [b["accuracy"] for b in bins if b["accuracy"] is not None]

plt.figure(figsize=(6, 5))
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration")
plt.plot(conf, acc, marker="o", label="ResidualStegNet")
plt.xlabel("Confidence")
plt.ylabel("Accuracy")
plt.title("Reliability Diagram")
plt.legend()
plt.tight_layout()
plt.savefig("figures/reliability_diagram.png", dpi=300)
plt.close()


# Figure 2: Robustness Results
rob = pd.read_csv("results/tables/table_robustness.csv")

plt.figure(figsize=(8, 5))
plt.bar(rob["Transformation"], rob["f1"])
plt.xlabel("Transformation")
plt.ylabel("F1 Score")
plt.title("Robustness Under Forensic Transformations")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("figures/robustness_f1.png", dpi=300)
plt.close()


# Figure 3: Triage Ranking Curve
triage = load_json("results/uncertainty_triage_outputs.json")

labels = [x["true_label"] for x in triage]
cumulative_stego = []
count = 0

for label in labels:
    if label == 1:
        count += 1
    cumulative_stego.append(count)

plt.figure(figsize=(7, 5))
plt.plot(range(1, len(cumulative_stego) + 1), cumulative_stego)
plt.xlabel("Ranked Images Inspected")
plt.ylabel("Cumulative Stego Images Found")
plt.title("Evidence Triage Ranking Curve")
plt.tight_layout()
plt.savefig("figures/triage_ranking_curve.png", dpi=300)
plt.close()


print("Figures saved:")
print("figures/reliability_diagram.png")
print("figures/robustness_f1.png")
print("figures/triage_ranking_curve.png")