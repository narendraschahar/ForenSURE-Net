"""
train_forensure.py — Clean training script for ForenSURE-Net

Usage:
  python scripts/train_forensure.py --config configs/forensure_verify.json
  python scripts/train_forensure.py --config configs/forensure_verify.json --debug  # 500 pairs only
"""

import argparse
import json
import sys
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.forensure_net import ForenSURENet
from src.datasets.bossbase_dataset import BOSSBasePairedDataset


# ── Collate ────────────────────────────────────────────────────────────────────

def paired_collate(batch):
    images = torch.cat([item[0] for item in batch], dim=0)
    labels = torch.cat([item[1] for item in batch], dim=0)
    return images, labels


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Val", leave=False, unit="batch"):
            logits = model(images.to(device)).squeeze(1)
            probs  = torch.sigmoid(logits).cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels.tolist())

    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    auc   = roc_auc_score(all_labels, all_probs)
    acc   = accuracy_score(all_labels, preds)
    return auc, acc


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--debug", action="store_true", help="Use only 500 pairs")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Data
    cover_dir = Path(cfg["cover_dir"])
    stego_dir = Path(cfg["stego_dir"])
    names = sorted([p.name for p in cover_dir.glob("*.pgm") if (stego_dir / p.name).exists()])
    random.seed(cfg.get("split_seed", 42))
    random.shuffle(names)

    if args.debug:
        names = names[:500]
        print(f"DEBUG: Using {len(names)} pairs")

    split = int(len(names) * 0.85)
    train_names, val_names = names[:split], names[split:]
    print(f"Train pairs: {len(train_names)} | Val pairs: {len(val_names)}")

    image_size  = cfg.get("image_size", 256)
    batch_size  = cfg.get("batch_size", 16)

    train_ds = BOSSBasePairedDataset(cover_dir, stego_dir, train_names, image_size=image_size, is_train=True)
    val_ds   = BOSSBasePairedDataset(cover_dir, stego_dir, val_names,   image_size=image_size, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, persistent_workers=True, collate_fn=paired_collate)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, persistent_workers=True, collate_fn=paired_collate)

    # Model
    model = ForenSURENet(dropout_p=cfg.get("dropout", 0.3)).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # Optimizer — SGD + momentum (best for steganalysis per literature)
    lr        = cfg.get("learning_rate", 0.01)
    epochs    = cfg.get("epochs", 30)
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, momentum=0.9, weight_decay=5e-4
    )
    # ReduceLROnPlateau: only drops LR when val AUC stops improving (patience=5).
    # Much better than StepLR which aggressively decays every N epochs.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )
    criterion = nn.BCEWithLogitsLoss()

    # Output
    save_dir = Path("experiments/checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / cfg.get("save_name", "forensure.pth")
    best_auc  = 0.0
    history   = []

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            logits = model(images).squeeze(1)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            cur_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{cur_lr:.5f}")

        val_auc, val_acc = evaluate(model, val_loader, device)
        elapsed = (time.time() - t0) / 60
        scheduler.step(val_auc)   # ReduceLROnPlateau monitors val_auc

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch [{epoch}/{epochs}] — {elapsed:.1f} min  |  LR: {cur_lr:.6f}")
        print(f"  Train Loss : {total_loss / len(train_loader):.4f}")
        print(f"  Val AUC    : {val_auc:.4f}  |  Val Acc: {val_acc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ Best model saved (AUC={best_auc:.4f})")

        history.append({"epoch": epoch, "loss": total_loss / len(train_loader),
                        "val_auc": val_auc, "val_acc": val_acc})

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    import json as _json
    (results_dir / (cfg.get("save_name", "forensure") + "_history.json")).write_text(
        _json.dumps(history, indent=2)
    )
    print(f"\nBest Val AUC: {best_auc:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
