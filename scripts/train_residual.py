import sys
import json
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
    device = get_device()
    print("Using device:", device)

    with open("configs/lsb_residual_config.json", "r") as f:
        config = json.load(f)

    epochs = config.get("epochs", 30)
    batch_size = config.get("batch_size", 16)
    lr = config.get("learning_rate", 1e-4)

    train_dataset = BOSSBaseDataset("data/splits/train.txt", image_size=256, is_train=True)
    val_dataset = BOSSBaseDataset("data/splits/val.txt", image_size=256, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ResidualStegNet(dropout_rate=config.get("dropout", 0.3)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Scheduler: Reduce learning rate when validation F1 stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

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