"""Depth-aware BEV attention used for backward refinement."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DepthAwareAttentionLite(nn.Module):
    """Compact depth-aware attention approximation in pure PyTorch."""

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

        query = self.query_proj(bev)
        key = self.key_proj(context_bev)
        value = self.value_proj(context_bev)
        similarity = (query * key).sum(dim=1, keepdim=True) / (query.shape[1] ** 0.5)
        weights = torch.sigmoid(similarity) * depth_weight.clamp(min=0.0, max=1.0)
        fused = query + weights * value
        out = self.out_proj(F.relu(fused, inplace=True))
        return F.relu(self.norm(out + bev), inplace=True)
