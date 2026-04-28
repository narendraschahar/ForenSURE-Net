import sys
import json
import argparse
import random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets.bossbase_dataset import BOSSBaseDataset
from src.models.residual_stegnet import ResidualStegNet
from src.training.trainer import Trainer

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/lsb_residual_config.json")
    args = parser.parse_args()

    device = get_device()
    print("Using device:", device)
    print("Loading config from:", args.config)

    with open(args.config, "r") as f:
        config = json.load(f)

    epochs = config.get("epochs", 30)
    batch_size = config.get("batch_size", 16)
    lr = config.get("learning_rate", 1e-4)

    # Dynamic Dataset Loading for Kaggle/Local Compatibility
    cover_dir = Path(config["cover_dir"])
    stego_dir = Path(config["stego_dir"])

    print(f"Scanning for covers in: {cover_dir}")
    print(f"Scanning for stegos in: {stego_dir}")

    cover_images = list(cover_dir.glob("*.pgm"))
    stego_images = list(stego_dir.glob("*.pgm"))

    if not cover_images or not stego_images:
        raise ValueError("Could not find images in the specified directories. Check Kaggle paths!")

    print(f"Found {len(cover_images)} covers and {len(stego_images)} stegos.")

    min_count = min(len(cover_images), len(stego_images))
    cover_images = cover_images[:min_count]
    stego_images = stego_images[:min_count]

    samples = [(str(p), 0) for p in cover_images] + [(str(p), 1) for p in stego_images]
    
    # Use consistent seed for splitting
    seed = config.get("split_seed", 42)
    random.seed(seed)
    random.shuffle(samples)

    # 85/15 train/val split
    split_idx = int(len(samples) * 0.85)
    train_split = samples[:split_idx]
    val_split = samples[split_idx:]

    print(f"Train split: {len(train_split)} images")
    print(f"Val split: {len(val_split)} images")

    train_dataset = BOSSBaseDataset(train_split, image_size=256, is_train=True)
    val_dataset = BOSSBaseDataset(val_split, image_size=256, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ResidualStegNet(dropout_rate=config.get("dropout", 0.3)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Scheduler: Reduce learning rate when validation F1 stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device, scheduler=scheduler)
    
    print(f"Starting training for {epochs} epochs...")
    trainer.train(
        epochs=epochs,
        save_name="residual_stegnet_best.pth",
        save_best=True,
        history_file="results/residual_stegnet_train_history.json"
    )
    print("\\nTraining completed.")

if __name__ == "__main__":
    main()