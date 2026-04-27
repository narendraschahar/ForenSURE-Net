import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from src.datasets.bossbase_dataset import BOSSBaseDataset
from src.models.residual_stegnet import ResidualStegNet
from src.evaluation.evaluator import evaluate_with_calibration

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    device = get_device()
    print("Using device:", device)

    test_dataset = BOSSBaseDataset("data/splits/test.txt", image_size=256)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

    model = ResidualStegNet().to(device)
    model.load_state_dict(torch.load("experiments/checkpoints/residual_stegnet_best.pth", map_location=device))

    results, all_labels, preds = evaluate_with_calibration(model, test_loader, device)

    print("\\nResidualStegNet Test Results with Calibration Metrics")
    print(json.dumps(results, indent=4))

    print("\\nClassification Report")
    print(classification_report(all_labels, preds, zero_division=0))

    Path("results").mkdir(parents=True, exist_ok=True)
    with open("results/residual_stegnet_test_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\\nSaved to results/residual_stegnet_test_results.json")

if __name__ == "__main__":
    main()