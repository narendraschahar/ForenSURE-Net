import json
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    preds = [1 if p >= 0.5 else 0 for p in all_probs]

    metrics = {
        "accuracy": accuracy_score(all_labels, preds),
        "precision": precision_score(all_labels, preds, zero_division=0),
        "recall": recall_score(all_labels, preds, zero_division=0),
        "f1": f1_score(all_labels, preds, zero_division=0),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        metrics["roc_auc"] = None

    return metrics, sum(all_probs)/len(all_probs) if all_probs else 0


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, scheduler=None, save_dir="experiments/checkpoints"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def train(self, epochs, save_name="model.pth", save_best=False, history_file=None):
        best_f1 = -1.0
        history = []

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0

            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images).squeeze(1)
                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            val_metrics, _ = evaluate_model(self.model, self.val_loader, self.device)

            if self.scheduler:
                # Assuming ReduceLROnPlateau based on val_f1
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["f1"])
                else:
                    self.scheduler.step()

            record = {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                **val_metrics
            }
            history.append(record)

            print(f"\\nEpoch [{epoch+1}/{epochs}]")
            print(f"Train Loss: {avg_loss:.4f}")
            print("Validation:", val_metrics)

            if save_best:
                if val_metrics["f1"] > best_f1:
                    best_f1 = val_metrics["f1"]
                    torch.save(self.model.state_dict(), self.save_dir / save_name)
                    print("Best model saved.")
            else:
                torch.save(self.model.state_dict(), self.save_dir / save_name)

        if history_file:
            Path(history_file).parent.mkdir(parents=True, exist_ok=True)
            with open(history_file, "w") as f:
                json.dump(history, f, indent=4)
            print(f"History saved to {history_file}")

        return history
