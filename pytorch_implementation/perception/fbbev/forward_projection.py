"""Pure-PyTorch forward projection from image-view features to BEV volume."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import FBBEVForwardConfig


class ForwardProjectionLite(nn.Module):
    """Lightweight LSS-style lift-splat projection with pure torch ops."""

    def __init__(self, cfg: FBBEVForwardConfig) -> None:
        super().__init__()
        self.bev_h = cfg.bev_h
        self.bev_w = cfg.bev_w
        self.bev_z = cfg.bev_z
        self.embed_dims = cfg.embed_dims

    def forward(
        self,
        context: torch.Tensor,
        depth_prob: torch.Tensor,
    ) -> torch.Tensor:
        """Build a BEV volume by depth weighting then trilinear resampling.

        Args:
            context: [B, Ncam, C, H, W]
            depth_prob: [B, Ncam, D, H, W]
        Returns:
            bev_volume: [B, C, Hbev, Wbev, Zbev]
        """

        if context.dim() != 5 or depth_prob.dim() != 5:
            raise ValueError("Expected context/depth with shape [B, Ncam, *, H, W].")
        if context.shape[0] != depth_prob.shape[0] or context.shape[1] != depth_prob.shape[1]:
            raise ValueError("Context/depth batch and camera dimensions must match.")

        volume = context.unsqueeze(2) * depth_prob.unsqueeze(3)  # [B, Ncam, D, C, H, W]
        volume = volume.mean(dim=1)  # [B, D, C, H, W]
        volume = volume.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, D, H, W]
        volume = F.interpolate(
            volume,
            size=(self.bev_z, self.bev_h, self.bev_w),
            mode="trilinear",
            align_corners=False,
        )
        return volume.permute(0, 1, 3, 4, 2).contiguous()
