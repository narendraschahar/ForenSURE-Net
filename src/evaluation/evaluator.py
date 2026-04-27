import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from src.calibration.metrics import expected_calibration_error, brier_score, reliability_bins


def evaluate_with_calibration(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for images, labels in loader:
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
        "confusion_matrix": confusion_matrix(all_labels, preds).tolist(),
        "ece": expected_calibration_error(all_labels, all_probs, n_bins=10),
        "brier_score": brier_score(all_labels, all_probs),
        "reliability_bins": reliability_bins(all_labels, all_probs, n_bins=10)
    }

    try:
        results["roc_auc"] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        results["roc_auc"] = None

    return results, all_labels, preds
