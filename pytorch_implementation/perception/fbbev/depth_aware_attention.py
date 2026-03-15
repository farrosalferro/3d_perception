"""Depth-aware BEV attention used for backward refinement."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DepthAwareAttentionLite(nn.Module):
    """Compact depth-aware attention with explicit depth-gating contracts."""

    def __init__(self, embed_dims: int) -> None:
        super().__init__()
        self.query_proj = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)
        self.key_proj = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)
        self.value_proj = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)
        self.out_proj = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)
        self.norm = nn.BatchNorm2d(embed_dims)

    def forward(
        self,
        bev: torch.Tensor,
        context_bev: torch.Tensor,
        depth_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
        bev: [B, C, H, W]
        context_bev: [B, C, H, W]
        depth_weight: [B, 1, H, W]
        """

        if bev.dim() != 4 or context_bev.dim() != 4 or depth_weight.dim() != 4:
            raise ValueError("DepthAwareAttentionLite expects bev/context/depth_weight 4D tensors.")
        if bev.shape != context_bev.shape:
            raise ValueError(f"bev/context_bev must match exactly, got {tuple(bev.shape)} vs {tuple(context_bev.shape)}")
        if depth_weight.shape[0] != bev.shape[0] or depth_weight.shape[-2:] != bev.shape[-2:]:
            raise ValueError(
                "depth_weight must align with batch/spatial dims of bev; "
                f"got depth_weight={tuple(depth_weight.shape)} bev={tuple(bev.shape)}"
            )
        if depth_weight.shape[1] != 1:
            raise ValueError(f"depth_weight channel dim must be 1, got {depth_weight.shape[1]}")
        if not torch.isfinite(bev).all() or not torch.isfinite(context_bev).all() or not torch.isfinite(depth_weight).all():
            raise ValueError("DepthAwareAttentionLite inputs must be finite.")

        depth_weight = depth_weight.clamp(min=0.0, max=1.0)
        query = self.query_proj(bev)
        key = self.key_proj(context_bev)
        value = self.value_proj(context_bev)
        similarity = (query * key).sum(dim=1, keepdim=True) / (query.shape[1] ** 0.5)
        weights = torch.sigmoid(similarity) * depth_weight
        fused = query + weights * value
        out = self.out_proj(F.relu(fused, inplace=True))
        return F.relu(self.norm(out + bev), inplace=True)
