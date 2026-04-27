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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ResidualStegNet(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.hpf = FixedHPF()

        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = ResidualBlock(32, 32, stride=1)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 128, stride=2)
        self.layer4 = ResidualBlock(128, 256, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.hpf(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)