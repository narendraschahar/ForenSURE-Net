"""
forensure_net.py — ForenSURE-Net Backbone

Architecture:
  FixedHPF (30 SRM filters, frozen)
  → abs()
  → 3× [Conv → GroupNorm → ReLU → (optional) AvgPool]
  → GlobalAvgPool
  → MC Dropout
  → Linear → sigmoid

Returns (logit, uncertainty) where:
  - logit      : raw classification score (use BCEWithLogitsLoss)
  - uncertainty: variance across N MC Dropout forward passes (0 at training time)
"""

import torch
import torch.nn as nn
import numpy as np


# ── SRM Filter Bank ────────────────────────────────────────────────────────────

def _build_srm_filters() -> torch.Tensor:
    """30 zero-sum high-pass filters from the Spatial Rich Model (Fridrich 2012)."""
    kernels = [
        np.array([[0,0,0,0,0],[0,0,0,0,0],[0,-1,2,-1,0],[0,0,0,0,0],[0,0,0,0,0]], np.float32) / 2,
        np.array([[0,0,0,0,0],[0,0,-1,0,0],[0,0,2,0,0],[0,0,-1,0,0],[0,0,0,0,0]], np.float32) / 2,
        np.array([[0,0,0,0,0],[0,-1,0,0,0],[0,0,2,0,0],[0,0,0,-1,0],[0,0,0,0,0]], np.float32) / 2,
        np.array([[0,0,0,0,0],[0,0,0,-1,0],[0,0,2,0,0],[0,-1,0,0,0],[0,0,0,0,0]], np.float32) / 2,
        np.array([[0,0,0,0,0],[0,0,1,0,0],[0,1,-4,1,0],[0,0,1,0,0],[0,0,0,0,0]], np.float32) / 4,
        np.array([[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]], np.float32) / 4,
        np.array([[0,0,0,0,0],[0,1,-2,1,0],[0,-2,4,-2,0],[0,1,-2,1,0],[0,0,0,0,0]], np.float32) / 4,
        np.array([[-1,-2,-2,-2,-1],[-2,4,4,4,-2],[-2,4,8,4,-2],[-2,4,4,4,-2],[-1,-2,-2,-2,-1]], np.float32) / 8,
        np.array([[0,0,0,0,0],[0,0,0,0,0],[-1,0,2,0,-1],[0,0,0,0,0],[0,0,0,0,0]], np.float32) / 2,
        np.array([[0,0,-1,0,0],[0,0,0,0,0],[0,0,2,0,0],[0,0,0,0,0],[0,0,-1,0,0]], np.float32) / 2,
    ]

    filters = np.zeros((30, 1, 5, 5), np.float32)
    for i, k in enumerate(kernels):
        filters[i, 0] = k

    rng = np.random.default_rng(42)
    for i in range(10, 30):
        base = kernels[i % len(kernels)].copy()
        rot  = np.rot90(base, k=(i // len(kernels)))
        noise = rng.standard_normal((5, 5)).astype(np.float32) * 0.005
        f = rot + noise
        f -= f.mean()          # enforce zero-sum
        filters[i, 0] = f

    filters -= filters.mean(axis=(2, 3), keepdims=True)
    return torch.FloatTensor(filters)


# ── Building Block ─────────────────────────────────────────────────────────────

class ConvGNReLU(nn.Module):
    """Conv2d → GroupNorm → ReLU. GroupNorm is MPS-safe and batch-size-independent."""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, pool=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, bias=False),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AvgPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.block(x))


# ── ForenSURE-Net ──────────────────────────────────────────────────────────────

class ForenSURENet(nn.Module):
    """
    ForenSURE-Net: Calibrated, Uncertainty-Aware Steganalysis Network.

    At TRAINING time  : forward() returns (logit,) — standard BCE loss.
    At INFERENCE time : call predict_with_uncertainty(x, n_passes=50) to get
                        (P_stego, uncertainty) for the forensic triage formula:
                            Triage = P_stego × (1 − uncertainty)
    """

    def __init__(self, dropout_p: float = 0.3):
        super().__init__()

        # ── Fixed SRM High-Pass Filter Bank ───────────────────────────────────
        self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
        with torch.no_grad():
            self.hpf.weight.data = _build_srm_filters()
        self.hpf.weight.requires_grad = False   # frozen — never update HPF

        # ── Feature Extraction ─────────────────────────────────────────────────
        self.features = nn.Sequential(
            ConvGNReLU(30,  64, kernel=3, pool=True),   # 256→128 (or 512→256)
            ConvGNReLU(64, 128, kernel=3, pool=True),   # 128→64
            ConvGNReLU(128, 256, kernel=3, pool=False),
        )

        # ── Global pooling + classifier ────────────────────────────────────────
        self.gap     = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_p)          # MC Dropout — kept ON at inference
        self.fc      = nn.Linear(256, 1)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass. Returns raw logit [B, 1]."""
        x = torch.abs(self.hpf(x))     # HPF residuals, take magnitude
        x = self.features(x)
        x = self.gap(x).flatten(1)     # [B, 256]
        x = self.dropout(x)
        return self.fc(x)              # [B, 1]  — raw logit

    # ------------------------------------------------------------------
    def predict_with_uncertainty(
        self, x: torch.Tensor, n_passes: int = 50
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Monte-Carlo Dropout inference.
        Keeps Dropout ACTIVE (train mode) during n_passes forward passes,
        then computes mean prediction and variance (= uncertainty).

        Returns:
            p_stego     : [B] — mean predicted probability of steganography
            uncertainty : [B] — variance across MC passes (higher = less confident)
        """
        self.eval()
        # Enable dropout during inference (MC Dropout trick)
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        probs = []
        with torch.no_grad():
            for _ in range(n_passes):
                logit = self.forward(x).squeeze(1)   # [B]
                probs.append(torch.sigmoid(logit))

        probs = torch.stack(probs, dim=0)             # [n_passes, B]
        p_stego     = probs.mean(dim=0)               # [B]
        uncertainty = probs.var(dim=0)                # [B]
        return p_stego, uncertainty
