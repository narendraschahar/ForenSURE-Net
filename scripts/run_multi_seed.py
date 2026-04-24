import json
import shutil
import subprocess
from pathlib import Path
import pandas as pd

seeds = [42, 123, 999]
all_results = []

Path("results/multi_seed").mkdir(parents=True, exist_ok=True)

for seed in seeds:
    print(f"\n==============================")
    print(f"Running experiment with seed {seed}")
    print(f"==============================")

    subprocess.run(["python", "scripts/create_splits.py", "--seed", str(seed)], check=True)

    checkpoint = Path("experiments/checkpoints/residual_stegnet_best.pth")
    if checkpoint.exists():
        checkpoint.unlink()

    subprocess.run(["python", "scripts/train_residual.py"], check=True)
    subprocess.run(["python", "scripts/evaluate_residual.py"], check=True)
    subprocess.run(["python", "scripts/calibrate_residual.py"], check=True)
    subprocess.run(["python", "scripts/evaluate_uncertainty.py"], check=True)
    subprocess.run(["python", "scripts/evaluate_case_folder.py"], check=True)

    with open("results/residual_stegnet_test_results.json") as f:
        test = json.load(f)

    with open("results/residual_temperature_scaling_results.json") as f:
        cal = json.load(f)

    with open("results/case_folder_metrics.json") as f:
        case = json.load(f)

    row = {
        "seed": seed,
        "accuracy": test.get("accuracy"),
        "precision": test.get("precision"),
        "recall": test.get("recall"),
        "f1": test.get("f1"),
        "roc_auc": test.get("roc_auc"),
        "ece": test.get("ece"),
        "brier_score": test.get("brier_score"),
        "calibrated_ece": cal.get("calibrated_ece"),
        "calibrated_brier": cal.get("calibrated_brier"),
        "top5_hit_rate": case.get("top5_hit_rate"),
        "top10_hit_rate": case.get("top10_hit_rate"),
        "mean_rank": case.get("mean_rank")
    }

    all_results.append(row)

    seed_dir = Path("results/multi_seed") / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    for file_name in [
        "residual_stegnet_test_results.json",
        "residual_temperature_scaling_results.json",
        "uncertainty_triage_outputs.json",
        "case_folder_metrics.json"
    ]:
        src = Path("results") / file_name
        if src.exists():
            shutil.copy2(src, seed_dir / file_name)

df = pd.DataFrame(all_results)
df.to_csv("results/multi_seed_summary.csv", index=False)

summary = df.drop(columns=["seed"]).agg(["mean", "std"])
summary.to_csv("results/multi_seed_mean_std.csv")

print("\nMulti-seed summary:")
print(df)

print("\nMean ± Std:")
print(summary)

print("\nSaved:")
print("results/multi_seed_summary.csv")
print("results/multi_seed_mean_std.csv")