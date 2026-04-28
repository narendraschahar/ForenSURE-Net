import torch
import torch.nn as nn
import torch.nn.functional as F

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
    Uses unpooled initial layers to capture high-frequency steganographic noise.
    """
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        
        # SOTA Learned SRM Filters (30 filters, 5x5)
        # Using learned filters achieves higher accuracy than fixed SRM.
        self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
        
        # Unpooled Layers (Crucial for noise preservation)
        self.layer1 = SRNetBlockType1(30, 32)
        self.layer2 = SRNetBlockType1(32, 32)
        
        self.layer3 = SRNetBlockType2(32, 32)
        self.layer4 = SRNetBlockType2(32, 32)
        self.layer5 = SRNetBlockType2(32, 32)
        self.layer6 = SRNetBlockType2(32, 32)
        self.layer7 = SRNetBlockType2(32, 32)
        
        # Pooled Layers (Spatial reduction)
        self.layer8 = SRNetBlockType3(32, 64)
        self.layer9 = SRNetBlockType3(64, 128)
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