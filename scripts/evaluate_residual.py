import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.datasets.bossbase_dataset import BOSSBaseDataset
from src.models.residual_stegnet import ResidualStegNet


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
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    model = ResidualStegNet().to(device)
    model.load_state_dict(
        torch.load(
            "experiments/checkpoints/residual_stegnet_best.pth",
            map_location=device
        )
    )
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits)

            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    preds = [1 if p >= 0.5 else 0 for p in all_probs]

    results = {
        "accuracy": accuracy_score(all_labels, preds),
        "precision": precision_score(all_labels, preds, zero_division=0),
        "recall": recall_score(all_labels, preds, zero_division=0),
        "f1": f1_score(all_labels, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(all_labels, preds).tolist()
    }

    try:
        results["roc_auc"] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        results["roc_auc"] = None

    print("\nResidualStegNet Test Results")
    print(json.dumps(results, indent=4))

    print("\nClassification Report")
    print(classification_report(all_labels, preds, zero_division=0))

    Path("results").mkdir(parents=True, exist_ok=True)

    with open("results/residual_stegnet_test_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nSaved to results/residual_stegnet_test_results.json")


if __name__ == "__main__":
    main()