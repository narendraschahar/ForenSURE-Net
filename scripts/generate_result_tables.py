import json
import pandas as pd
from pathlib import Path

Path("results/tables").mkdir(parents=True, exist_ok=True)

files = {
    "test": "results/residual_stegnet_test_results.json",
    "calibration": "results/residual_temperature_scaling_results.json",
    "case_folder": "results/case_folder_metrics.json",
    "robustness": "results/robustness_results.json"
}

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

# Table 1: Test performance
test = load_json(files["test"])
test_table = pd.DataFrame([{
    "Accuracy": test.get("accuracy"),
    "Precision": test.get("precision"),
    "Recall": test.get("recall"),
    "F1": test.get("f1"),
    "ROC-AUC": test.get("roc_auc"),
    "ECE": test.get("ece"),
    "Brier Score": test.get("brier_score")
}])
test_table.to_csv("results/tables/table_detection_calibration.csv", index=False)

# Table 2: Temperature scaling
cal = load_json(files["calibration"])
cal_table = pd.DataFrame([{
    "Temperature": cal.get("temperature"),
    "Raw ECE": cal.get("raw_ece"),
    "Calibrated ECE": cal.get("calibrated_ece"),
    "Raw Brier": cal.get("raw_brier"),
    "Calibrated Brier": cal.get("calibrated_brier")
}])
cal_table.to_csv("results/tables/table_temperature_scaling.csv", index=False)

# Table 3: Case-folder triage
case = load_json(files["case_folder"])
case_table = pd.DataFrame([case])
case_table.to_csv("results/tables/table_case_folder_triage.csv", index=False)

# Table 4: Robustness
rob = load_json(files["robustness"])
rob_table = pd.DataFrame.from_dict(rob, orient="index").reset_index()
rob_table = rob_table.rename(columns={"index": "Transformation"})
rob_table.to_csv("results/tables/table_robustness.csv", index=False)

print("Tables generated in results/tables/")
print(test_table)
print(cal_table)
print(case_table)
print(rob_table)