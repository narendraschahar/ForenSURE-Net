import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src.datasets.bossbase_dataset import BOSSBaseDataset
from src.models.residual_stegnet import ResidualStegNet
from src.calibration.temperature_scaling import TemperatureScaler
from src.calibration.metrics import expected_calibration_error, brier_score


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collect_logits(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images).squeeze(1)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_logits), torch.cat(all_labels)


def main():
    device = get_device()
    print("Using device:", device)

    val_dataset = BOSSBaseDataset("data/splits/val.txt", image_size=256)
    test_dataset = BOSSBaseDataset("data/splits/test.txt", image_size=256)

    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

    model = ResidualStegNet().to(device)
    model.load_state_dict(
        torch.load(
            "experiments/checkpoints/residual_stegnet_best.pth",
            map_location=device
        )
    )

    val_logits, val_labels = collect_logits(model, val_loader, device)

    scaler = TemperatureScaler()
    scaler.fit(val_logits, val_labels)

    print("Learned temperature:", scaler.temperature.item())

    test_logits, test_labels = collect_logits(model, test_loader, device)

    raw_probs = torch.sigmoid(test_logits).numpy()
    calibrated_probs = torch.sigmoid(scaler(test_logits)).detach().numpy()
    labels = test_labels.numpy()

    results = {
        "temperature": float(scaler.temperature.item()),
        "raw_ece": expected_calibration_error(labels, raw_probs, n_bins=10),
        "calibrated_ece": expected_calibration_error(labels, calibrated_probs, n_bins=10),
        "raw_brier": brier_score(labels, raw_probs),
        "calibrated_brier": brier_score(labels, calibrated_probs)
    }

    Path("results").mkdir(parents=True, exist_ok=True)
    Path("experiments/calibration").mkdir(parents=True, exist_ok=True)

    with open("results/residual_temperature_scaling_results.json", "w") as f:
        json.dump(results, f, indent=4)

    torch.save(
        {"temperature": scaler.temperature.detach().cpu()},
        "experiments/calibration/residual_temperature_scaler.pth"
    )

    print(json.dumps(results, indent=4))
    print("Saved calibration results and temperature scaler.")


if __name__ == "__main__":
    main()