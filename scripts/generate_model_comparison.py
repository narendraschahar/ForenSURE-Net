import json
import pandas as pd
from pathlib import Path

Path("results/tables").mkdir(parents=True, exist_ok=True)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

baseline = load_json("results/baseline_test_results.json") if Path("results/baseline_test_results.json").exists() else {}
residual = load_json("results/residual_stegnet_test_results.json")
cal = load_json("results/residual_temperature_scaling_results.json")
case = load_json("results/case_folder_metrics.json")

rows = []

if baseline:
    rows.append({
        "Model": "BaselineCNN",
        "High-pass": "No",
        "Calibration": "No",
        "Uncertainty": "No",
        "Triage": "No",
        "Accuracy": baseline.get("accuracy"),
        "F1": baseline.get("f1"),
        "ROC-AUC": baseline.get("roc_auc"),
        "ECE": baseline.get("ece"),
        "Top-10 Hit Rate": "-"
    })

rows.append({
    "Model": "ResidualStegNet",
    "High-pass": "Yes",
    "Calibration": "No",
    "Uncertainty": "No",
    "Triage": "No",
    "Accuracy": residual.get("accuracy"),
    "F1": residual.get("f1"),
    "ROC-AUC": residual.get("roc_auc"),
    "ECE": residual.get("ece"),
    "Top-10 Hit Rate": "-"
})

rows.append({
    "Model": "ForenSURE-Net Full",
    "High-pass": "Yes",
    "Calibration": "Yes",
    "Uncertainty": "Yes",
    "Triage": "Yes",
    "Accuracy": residual.get("accuracy"),
    "F1": residual.get("f1"),
    "ROC-AUC": residual.get("roc_auc"),
    "ECE": cal.get("calibrated_ece"),
    "Top-10 Hit Rate": case.get("top10_hit_rate")
})

df = pd.DataFrame(rows)
df.to_csv("results/tables/table_model_comparison.csv", index=False)

print(df)
print("Saved: results/tables/table_model_comparison.csv")