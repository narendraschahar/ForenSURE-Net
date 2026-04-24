import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedHPF(nn.Module):
    def __init__(self):
        super().__init__()

        kernels = torch.tensor([
            [[0, -1, 0],
             [-1, 4, -1],
             [0, -1, 0]],

            [[-1, 2, -1],
             [2, -4, 2],
             [-1, 2, -1]],

            [[1, -2, 1],
             [-2, 4, -2],
             [1, -2, 1]]
        ], dtype=torch.float32)

        self.weight = nn.Parameter(
            kernels.unsqueeze(1),
            requires_grad=False
        )

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=1)


class ResidualStegNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.hpf = FixedHPF()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.hpf(x)
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)