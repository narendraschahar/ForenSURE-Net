import sys
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

    train_dataset = BOSSBaseDataset("data/splits/train.txt", image_size=256)
    val_dataset = BOSSBaseDataset("data/splits/val.txt", image_size=256)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    model = ResidualStegNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)
    
    print("Starting training...")
    trainer.train(
        epochs=10,
        save_name="residual_stegnet_best.pth",
        save_best=True,
        history_file="results/residual_stegnet_train_history.json"
    )
    print("\\nTraining completed.")

if __name__ == "__main__":
    main()