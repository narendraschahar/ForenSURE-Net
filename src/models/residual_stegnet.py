import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _build_srm_filters():
    """
    Initialize 30 high-pass filters in the style of the Spatial Rich Model (SRM).
    All filters enforce a zero-sum constraint, which is the mathematical definition
    of a high-pass filter — it responds to noise, not image content.
    """
    filters = np.zeros((30, 1, 5, 5), dtype=np.float32)

    # ── Core SRM kernels ──────────────────────────────────────────────────────
    # These are canonical linear predictors from Fridrich & Kodovsky (2012).
    kernels = [
        # First-order linear predictors (horizontal / vertical / diagonal)
        np.array([[0,  0,  0,  0, 0],
                  [0,  0,  0,  0, 0],
                  [0, -1,  2, -1, 0],
                  [0,  0,  0,  0, 0],
                  [0,  0,  0,  0, 0]], dtype=np.float32) / 2,

        np.array([[0,  0,  0,  0, 0],
                  [0,  0, -1,  0, 0],
                  [0,  0,  2,  0, 0],
                  [0,  0, -1,  0, 0],
                  [0,  0,  0,  0, 0]], dtype=np.float32) / 2,

        np.array([[0,  0,  0,  0, 0],
                  [0, -1,  0,  0, 0],
                  [0,  0,  2,  0, 0],
                  [0,  0,  0, -1, 0],
                  [0,  0,  0,  0, 0]], dtype=np.float32) / 2,

        np.array([[0,  0,  0,  0, 0],
                  [0,  0,  0, -1, 0],
                  [0,  0,  2,  0, 0],
                  [0, -1,  0,  0, 0],
                  [0,  0,  0,  0, 0]], dtype=np.float32) / 2,

        # Second-order (Laplacian family)
        np.array([[ 0,  0,  0,  0,  0],
                  [ 0,  0,  1,  0,  0],
                  [ 0,  1, -4,  1,  0],
                  [ 0,  0,  1,  0,  0],
                  [ 0,  0,  0,  0,  0]], dtype=np.float32) / 4,

        np.array([[ 0,  0,  0,  0,  0],
                  [ 0, -1,  2, -1,  0],
                  [ 0,  2, -4,  2,  0],
                  [ 0, -1,  2, -1,  0],
                  [ 0,  0,  0,  0,  0]], dtype=np.float32) / 4,

        # Third-order
        np.array([[ 0,  0,  0,  0,  0],
                  [ 0,  1, -2,  1,  0],
                  [ 0, -2,  4, -2,  0],
                  [ 0,  1, -2,  1,  0],
                  [ 0,  0,  0,  0,  0]], dtype=np.float32) / 4,

        # 5x5 Laplacian
        np.array([[-1, -2, -2, -2, -1],
                  [-2,  4,  4,  4, -2],
                  [-2,  4,  8,  4, -2],
                  [-2,  4,  4,  4, -2],
                  [-1, -2, -2, -2, -1]], dtype=np.float32) / 8,

        # Horizontal gradient
        np.array([[ 0,  0,  0,  0,  0],
                  [ 0,  0,  0,  0,  0],
                  [-1,  0,  2,  0, -1],
                  [ 0,  0,  0,  0,  0],
                  [ 0,  0,  0,  0,  0]], dtype=np.float32) / 2,

        # Vertical gradient
        np.array([[ 0,  0, -1,  0,  0],
                  [ 0,  0,  0,  0,  0],
                  [ 0,  0,  2,  0,  0],
                  [ 0,  0,  0,  0,  0],
                  [ 0,  0, -1,  0,  0]], dtype=np.float32) / 2,
    ]

    # ── Fill first 10 slots with canonical kernels ─────────────────────────
    for i, k in enumerate(kernels):
        filters[i, 0] = k

    # ── Fill remaining 20 slots with rotations & small random perturbations ─
    rng = np.random.default_rng(42)
    for i in range(10, 30):
        base = kernels[i % len(kernels)].copy()
        rot  = np.rot90(base, k=(i // len(kernels)))
        noise = rng.standard_normal((5, 5)).astype(np.float32) * 0.005
        f = rot + noise
        # Enforce zero-sum (high-pass constraint)
        f -= f.mean()
        filters[i, 0] = f

    # Final zero-sum pass for all filters
    filters -= filters.mean(axis=(2, 3), keepdims=True)
    return torch.FloatTensor(filters)


class SRNetBlockType1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SRNetBlockType2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + res


class SRNetBlockType3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.pool(out)
        return out + res


class SRNetBlockType4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.pool(out)
        return out


class ResidualStegNet(nn.Module):
    """
    SRNet (Steganalysis Residual Network) Architecture.
    The first layer is initialized with 30 SRM high-pass filters (zero-sum constraint).
    This is the critical difference vs. random initialization: the network immediately
    sees steganographic noise residuals rather than image content.
    """
    def __init__(self, dropout_rate=0.3):
        super().__init__()

        # HPF: initialized with SRM-style zero-sum high-pass filters
        self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
        with torch.no_grad():
            self.hpf.weight.data = _build_srm_filters()

        # Unpooled Layers (Crucial for noise preservation)
        self.layer1 = SRNetBlockType1(30, 32)
        self.layer2 = SRNetBlockType1(32, 32)

        self.layer3 = SRNetBlockType2(32, 32)
        self.layer4 = SRNetBlockType2(32, 32)
        self.layer5 = SRNetBlockType2(32, 32)
        self.layer6 = SRNetBlockType2(32, 32)
        self.layer7 = SRNetBlockType2(32, 32)

        # Pooled Layers (Spatial reduction)
        self.layer8  = SRNetBlockType3(32, 64)
        self.layer9  = SRNetBlockType3(64, 128)
        self.layer10 = SRNetBlockType3(128, 256)

        # Final aggregation
        self.layer11 = SRNetBlockType4(256, 512)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.hpf(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)

        x = x.flatten(1)
        return self.classifier(x)