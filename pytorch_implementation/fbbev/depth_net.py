"""Depth/context prediction used by FB-BEV forward projection."""

from __future__ import annotations

import torch
from torch import nn

from .config import FBBEVForwardConfig


class FBBEVDepthNetLite(nn.Module):
    """Predict context features and per-pixel depth probabilities."""

    def __init__(self, cfg: FBBEVForwardConfig) -> None:
        super().__init__()
        channels = cfg.backbone_neck.out_channels
        self.embed_dims = cfg.embed_dims
        self.depth_bins = cfg.depth_bins
        self.trunk = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.context_proj = nn.Conv2d(channels, self.embed_dims, kernel_size=1)
        self.depth_logits = nn.Conv2d(channels, self.depth_bins, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
        feat: [B, Ncam, C, H, W]
        Returns:
            context: [B, Ncam, Cbev, H, W]
            depth_prob: [B, Ncam, D, H, W]
        """

        if feat.dim() != 5:
            raise ValueError(f"Expected feature shape [B, Ncam, C, H, W], got {tuple(feat.shape)}")
        batch_size, num_cams, channels, height, width = feat.shape
        flat = feat.reshape(batch_size * num_cams, channels, height, width)
        trunk = self.trunk(flat)
        context = self.context_proj(trunk)
        depth_prob = self.depth_logits(trunk).softmax(dim=1)
        context = context.view(batch_size, num_cams, self.embed_dims, height, width)
        depth_prob = depth_prob.view(batch_size, num_cams, self.depth_bins, height, width)
        return context, depth_prob
