import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src.datasets.bossbase_dataset import BOSSBaseDataset
from src.models.residual_stegnet import ResidualStegNet
from src.triage.triage_scorer import score_triage

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_temperature():
    ckpt = torch.load("experiments/calibration/residual_temperature_scaler.pth", map_location="cpu")
    return float(ckpt["temperature"].item())

def main():
    device = get_device()
    print("Using device:", device)

    dataset = BOSSBaseDataset("data/splits/test.txt", image_size=256)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    model = ResidualStegNet().to(device)
    model.load_state_dict(torch.load("experiments/checkpoints/residual_stegnet_best.pth", map_location=device))

    temperature = load_temperature()

    outputs = []
    idx_counter = 0

    for images, labels in loader:
        mean_prob, reliability, uncertainty, triage = score_triage(model, images, device, temperature)

        for i in range(len(mean_prob)):
            outputs.append({
                "index": idx_counter,
                "true_label": int(labels[i].item()),
                "stego_probability": float(mean_prob[i]),
                "reliability_score": float(reliability[i]),
                "uncertainty_score": float(uncertainty[i]),
                "triage_score": float(triage[i])
            })
            idx_counter += 1

    outputs = sorted(outputs, key=lambda x: x["triage_score"], reverse=True)

    Path("results").mkdir(parents=True, exist_ok=True)
    with open("results/uncertainty_triage_outputs.json", "w") as f:
        json.dump(outputs, f, indent=4)

    print("Saved:", "results/uncertainty_triage_outputs.json")
    print("Top 10 ranked suspicious images:\\n")

    for row in outputs[:10]:
        print(row)

if __name__ == "__main__":
    main()