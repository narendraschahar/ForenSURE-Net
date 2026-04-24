import torch
import torch.nn as nn


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature.clamp(min=1e-6)

    def fit(self, logits, labels, max_iter=100):
        logits = logits.detach()
        labels = labels.detach()

        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        criterion = nn.BCEWithLogitsLoss()

        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self