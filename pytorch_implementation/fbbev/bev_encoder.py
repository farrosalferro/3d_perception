"""BEV encoder blocks for FB-BEV."""

from __future__ import annotations

import torch
from torch import nn

from .config import FBBEVForwardConfig


class _BEVResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu(x + identity)
        return x


class BEVEncoderLite(nn.Module):
    """Compact BEV encoder used before detection head."""

    def __init__(self, cfg: FBBEVForwardConfig) -> None:
        super().__init__()
        channels = cfg.embed_dims
        self.blocks = nn.Sequential(
            _BEVResBlock(channels),
            _BEVResBlock(channels),
        )

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        if bev.dim() != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(bev.shape)}")
        return self.blocks(bev)
