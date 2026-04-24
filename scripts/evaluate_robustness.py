import sys
import json
from pathlib import Path
from io import BytesIO

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.datasets.bossbase_dataset import BOSSBaseDataset
from src.models.residual_stegnet import ResidualStegNet


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def apply_transform(img_tensor, transform_name):
    img = img_tensor.squeeze(0).numpy()
    img = ((img * 0.5 + 0.5) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img, mode="L")

    if transform_name == "jpeg_q75":
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=75)
        buffer.seek(0)
        pil_img = Image.open(buffer).convert("L")

    elif transform_name == "jpeg_q50":
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=50)
        buffer.seek(0)
        pil_img = Image.open(buffer).convert("L")

    elif transform_name == "resize_down_up":
        pil_img = pil_img.resize((128, 128), Image.BICUBIC)
        pil_img = pil_img.resize((256, 256), Image.BICUBIC)

    elif transform_name == "center_crop":
        pil_img = pil_img.crop((16, 16, 240, 240))
        pil_img = pil_img.resize((256, 256), Image.BICUBIC)

    elif transform_name == "gaussian_noise":
        arr = np.array(pil_img).astype(np.float32)
        noise = np.random.normal(0, 5, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr, mode="L")

    elif transform_name == "screenshot_like":
        pil_img = pil_img.resize((220, 220), Image.BICUBIC)
        pil_img = pil_img.resize((256, 256), Image.BICUBIC)
        pil_img = pil_img.filter(ImageFilter.SMOOTH)

    arr = np.array(pil_img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    tensor = torch.tensor(arr).unsqueeze(0).float()

    return tensor


def evaluate_transform(model, loader, device, transform_name):
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            transformed = []

            for img in images:
                transformed.append(apply_transform(img, transform_name))

            images = torch.stack(transformed).to(device)

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
    }

    try:
        results["roc_auc"] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        results["roc_auc"] = None

    return results


def main():
    device = get_device()
    print("Using device:", device)

    dataset = BOSSBaseDataset("data/splits/test.txt", image_size=256)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    model = ResidualStegNet().to(device)
    model.load_state_dict(
        torch.load(
            "experiments/checkpoints/residual_stegnet_best.pth",
            map_location=device
        )
    )

    transforms = [
        "jpeg_q75",
        "jpeg_q50",
        "resize_down_up",
        "center_crop",
        "gaussian_noise",
        "screenshot_like"
    ]

    final_results = {}

    for transform_name in transforms:
        print("\nEvaluating:", transform_name)
        final_results[transform_name] = evaluate_transform(
            model,
            loader,
            device,
            transform_name
        )
        print(final_results[transform_name])

    Path("results").mkdir(parents=True, exist_ok=True)

    with open("results/robustness_results.json", "w") as f:
        json.dump(final_results, f, indent=4)

    print("\nSaved to results/robustness_results.json")


if __name__ == "__main__":
    main()