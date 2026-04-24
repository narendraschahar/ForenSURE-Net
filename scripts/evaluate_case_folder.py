import json
import random
import numpy as np
from pathlib import Path

random.seed(42)

INPUT_FILE = "results/uncertainty_triage_outputs.json"
OUTPUT_FILE = "results/case_folder_metrics.json"

FOLDER_SIZE = 100
NUM_STEGO = 10
NUM_TRIALS = 50


def evaluate_trial(all_items):
    stego_items = [x for x in all_items if x["true_label"] == 1]
    cover_items = [x for x in all_items if x["true_label"] == 0]

    selected_stego = random.sample(stego_items, min(NUM_STEGO, len(stego_items)))
    selected_cover = random.sample(
        cover_items,
        FOLDER_SIZE - len(selected_stego)
    )

    folder = selected_stego + selected_cover
    folder = sorted(folder, key=lambda x: x["triage_score"], reverse=True)

    stego_ranks = []

    for rank, item in enumerate(folder, start=1):
        if item["true_label"] == 1:
            stego_ranks.append(rank)

    top5_hit = int(any(r <= 5 for r in stego_ranks))
    top10_hit = int(any(r <= 10 for r in stego_ranks))

    return {
        "top5_hit": top5_hit,
        "top10_hit": top10_hit,
        "mean_rank": float(np.mean(stego_ranks)),
        "median_rank": float(np.median(stego_ranks)),
        "best_rank": int(min(stego_ranks))
    }


def main():
    with open(INPUT_FILE, "r") as f:
        all_items = json.load(f)

    results = []

    for _ in range(NUM_TRIALS):
        results.append(evaluate_trial(all_items))

    summary = {
        "folder_size": FOLDER_SIZE,
        "num_stego": NUM_STEGO,
        "num_trials": NUM_TRIALS,
        "top5_hit_rate": float(np.mean([r["top5_hit"] for r in results])),
        "top10_hit_rate": float(np.mean([r["top10_hit"] for r in results])),
        "mean_rank": float(np.mean([r["mean_rank"] for r in results])),
        "median_rank": float(np.mean([r["median_rank"] for r in results])),
        "best_rank": float(np.mean([r["best_rank"] for r in results]))
    }

    Path("results").mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(summary, f, indent=4)

    print(json.dumps(summary, indent=4))


if __name__ == "__main__":
    main()