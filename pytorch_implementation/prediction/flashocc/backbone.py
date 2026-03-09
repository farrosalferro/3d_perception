"""Occupancy-to-BEV encoder for FlashOcc-style prediction."""

from __future__ import annotations

import torch
from torch import nn

from .config import FlashOccConfig


class ResidualConvBlock(nn.Module):
    """Small residual block used in BEV feature extraction."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + identity
        return self.act(x)


class FlashOccBEVEncoder(nn.Module):
    """Encode occupancy history [B, T, Cin, H, W] into BEV features."""

    def __init__(self, cfg: FlashOccConfig) -> None:
        super().__init__()
        bcfg = cfg.backbone
        self.stem = nn.Sequential(
            nn.Conv2d(
                bcfg.in_channels,
                bcfg.embed_dims,
                kernel_size=bcfg.stem_kernel,
                stride=bcfg.stem_stride,
                padding=bcfg.stem_padding,
                bias=False,
            ),
            nn.BatchNorm2d(bcfg.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([ResidualConvBlock(bcfg.embed_dims) for _ in range(bcfg.num_res_blocks)])

    def forward(self, occ_seq: torch.Tensor) -> torch.Tensor:
        if occ_seq.dim() != 5:
            raise ValueError(f"Expected occupancy sequence [B, T, Cin, H, W], got {tuple(occ_seq.shape)}")
        batch_size, hist, channels, height, width = occ_seq.shape
        x = occ_seq.reshape(batch_size * hist, channels, height, width)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        _, embed, bev_h, bev_w = x.shape
        return x.view(batch_size, hist, embed, bev_h, bev_w)

