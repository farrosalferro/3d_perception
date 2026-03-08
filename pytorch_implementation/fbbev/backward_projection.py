"""Backward projection refinement for FB-BEV."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import FBBEVForwardConfig
from .depth_aware_attention import DepthAwareAttentionLite


class BackwardProjectionLite(nn.Module):
    """Refine BEV features with camera context and depth confidence."""

    def __init__(self, cfg: FBBEVForwardConfig) -> None:
        super().__init__()
        self.depth_attention = DepthAwareAttentionLite(cfg.embed_dims)
        self.post = nn.Sequential(
            nn.Conv2d(cfg.embed_dims, cfg.embed_dims, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.embed_dims),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        bev_volume: torch.Tensor,
        context: torch.Tensor,
        depth_prob: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
        bev_volume: [B, C, Hbev, Wbev, Zbev]
        context: [B, Ncam, C, Hf, Wf]
        depth_prob: [B, Ncam, D, Hf, Wf]
        Returns:
            refined_bev: [B, C, Hbev, Wbev]
        """

        if bev_volume.dim() != 5:
            raise ValueError(f"Expected bev_volume [B, C, H, W, Z], got {tuple(bev_volume.shape)}")
        bev_2d = bev_volume.mean(dim=-1)
        context_bev = context.mean(dim=1)
        depth_weight = depth_prob.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        context_bev = F.interpolate(context_bev, size=bev_2d.shape[-2:], mode="bilinear", align_corners=False)
        depth_weight = F.interpolate(depth_weight, size=bev_2d.shape[-2:], mode="bilinear", align_corners=False)
        refined = self.depth_attention(bev_2d, context_bev, depth_weight)
        return self.post(refined)
