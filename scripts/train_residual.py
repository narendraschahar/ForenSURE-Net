import sys
import json
import argparse
import random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.datasets.bossbase_dataset import BOSSBasePairedDataset
from src.models.residual_stegnet import ResidualStegNet
from src.training.trainer import Trainer

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def paired_collate_fn(batch):
    images = torch.cat([item[0] for item in batch], dim=0)
    labels = torch.cat([item[1] for item in batch], dim=0)
    return images, labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to pre-trained weights for Curriculum Phase 2")
    parser.add_argument("--debug", action="store_true", help="Use only 500 pairs for quick sanity check")
    args = parser.parse_args()

    device = get_device()
    print("Using device:", device)
    print("Loading config from:", args.config)

    with open(args.config, "r") as f:
        config = json.load(f)

    epochs = config.get("epochs", 30)
    batch_size = config.get("batch_size", 16)
    lr = config.get("learning_rate", 1e-4)

    cover_dir = Path(config["cover_dir"])
    stego_dir = Path(config["stego_dir"])

    print(f"Scanning for covers in: {cover_dir}")
    print(f"Scanning for stegos in: {stego_dir}")

    # For paired dataset, we just need a list of filenames that exist in both
    cover_images = {p.name for p in cover_dir.glob("*.pgm")}
    stego_images = {p.name for p in stego_dir.glob("*.pgm")}
    
    valid_names = list(cover_images.intersection(stego_images))

    if not valid_names:
        raise ValueError("Could not find matching Cover/Stego pairs in the directories. Check Kaggle paths!")

    if args.debug:
        valid_names = valid_names[:500]
        print(f"--- DEBUG MODE: Using only {len(valid_names)} pairs for fast verification ---")

    print(f"Found {len(valid_names)} perfect Cover/Stego pairs.")

    # Use consistent seed for splitting
    seed = config.get("split_seed", 42)
    random.seed(seed)
    random.shuffle(valid_names)

    # 85/15 train/val split
    split_idx = int(len(valid_names) * 0.85)
    train_split = valid_names[:split_idx]
    val_split = valid_names[split_idx:]

    print(f"Train pairs: {len(train_split)} (Yields {len(train_split)*2} images per epoch)")
    print(f"Val pairs: {len(val_split)} (Yields {len(val_split)*2} images per eval)")

    train_dataset = BOSSBasePairedDataset(cover_dir, stego_dir, train_split, image_size=256, is_train=True)
    val_dataset = BOSSBasePairedDataset(cover_dir, stego_dir, val_split, image_size=256, is_train=False)

    # Note: If batch_size is 16, the model actually receives 32 images (16 covers + 16 stegos)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=paired_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=paired_collate_fn)

    model = ResidualStegNet(dropout_rate=config.get("dropout", 0.3)).to(device)

    # Resume from checkpoint if provided (Curriculum Phase 2)
    if args.resume:
        print(f"--- CURRICULUM LEARNING: Loading Phase 1 Weights from {args.resume} ---")
        checkpoint = torch.load(args.resume, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Scheduler: Reduce learning rate when validation F1 stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device, scheduler=scheduler)
    
    save_name = config.get("save_name", "residual_stegnet_best.pth")
    print(f"Starting training for {epochs} epochs...")
    trainer.train(
        epochs=epochs,
        save_name=save_name,
        save_best=True,
        history_file=f"results/{save_name}_history.json"
    )
    print("\\nTraining completed.")

if __name__ == "__main__":
    main()