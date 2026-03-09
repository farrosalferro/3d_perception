"""Temporal fusion for BEV occupancy features."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import FlashOccConfig


class FlashOccTemporalMixer(nn.Module):
    """Fuse history along time axis with lightweight temporal convolution."""

    def __init__(self, cfg: FlashOccConfig) -> None:
        super().__init__()
        embed_dims = cfg.backbone.embed_dims
        self.temporal_conv = nn.Conv1d(embed_dims, embed_dims, kernel_size=3, padding=1, bias=False)
        self.norm = nn.BatchNorm1d(embed_dims)
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.GELU(),
        )

    def forward(self, bev_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if bev_seq.dim() != 5:
            raise ValueError(f"Expected BEV sequence [B, T, C, H, W], got {tuple(bev_seq.shape)}")
        batch_size, hist, embed_dims, bev_h, bev_w = bev_seq.shape
        tokens = bev_seq.permute(0, 3, 4, 2, 1).reshape(batch_size * bev_h * bev_w, embed_dims, hist)
        tokens = self.temporal_conv(tokens)
        tokens = self.norm(tokens)
        tokens = F.gelu(tokens)
        last_step = tokens[..., -1].view(batch_size, bev_h, bev_w, embed_dims).permute(0, 3, 1, 2)
        fused = self.proj(last_step)
        return fused, tokens

