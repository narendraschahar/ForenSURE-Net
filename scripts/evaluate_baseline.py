import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from src.datasets.bossbase_dataset import BOSSBaseDataset
from src.models.baseline_cnn import BaselineCNN


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

    model = BaselineCNN().to(device)
    model.load_state_dict(torch.load("experiments/checkpoints/baseline_cnn.pth", map_location=device))
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

    print("Accuracy:", accuracy_score(all_labels, preds))
    print("Precision:", precision_score(all_labels, preds, zero_division=0))
    print("Recall:", recall_score(all_labels, preds, zero_division=0))
    print("F1:", f1_score(all_labels, preds, zero_division=0))

    try:
        print("ROC-AUC:", roc_auc_score(all_labels, all_probs))
    except ValueError:
        print("ROC-AUC: Not available")

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, preds))


if __name__ == "__main__":
    main()