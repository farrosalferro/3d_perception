"""Core Sparse4D blocks implemented in plain PyTorch."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class SparseBox3DEncoderLite(nn.Module):
    """Encodes box anchors into query-aligned embeddings."""

    def __init__(self, box_code_size: int, embed_dims: int) -> None:
        super().__init__()
        hidden = max(embed_dims // 2, 64)
        self.mlp = nn.Sequential(
            nn.Linear(box_code_size, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, embed_dims),
        )

    def forward(self, anchors: torch.Tensor) -> torch.Tensor:
        return self.mlp(anchors)


class DeformableFeatureAggregationLite(nn.Module):
    """Simple image-context aggregation surrogate for Sparse4D cross attention."""

    def __init__(self, embed_dims: int) -> None:
        super().__init__()
        self.context_proj = nn.Linear(embed_dims, embed_dims)
        self.query_proj = nn.Linear(embed_dims, embed_dims)
        self.norm = nn.LayerNorm(embed_dims)

    def forward(
        self,
        query: torch.Tensor,
        img_feats: Sequence[torch.Tensor],
        projection_mat: torch.Tensor | None = None,
        image_wh: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not img_feats:
            raise ValueError("img_feats must contain at least one feature level.")
        level_contexts = [feat.mean(dim=(1, 3, 4)) for feat in img_feats]
        context = torch.stack(level_contexts, dim=0).mean(dim=0)  # [B, C]
        context = self.context_proj(context).unsqueeze(1)  # [B, 1, C]

        if projection_mat is not None:
            proj_scale = projection_mat[..., :3, :3].abs().mean(dim=(1, 2, 3)).view(-1, 1, 1)
            context = context * (1.0 + 0.01 * proj_scale)
        if image_wh is not None:
            wh_scale = image_wh.float().mean(dim=(1, 2)).view(-1, 1, 1)
            context = context * (1.0 + 1e-4 * wh_scale)

        fused = self.query_proj(query) + context
        return self.norm(fused)


class SparseDecoderLayerLite(nn.Module):
    """Sparse4D decoder layer: self-attn + image aggregation + FFN."""

    def __init__(
        self,
        *,
        embed_dims: int,
        num_heads: int,
        ffn_dims: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = DeformableFeatureAggregationLite(embed_dims=embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, ffn_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dims, embed_dims),
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

    def forward(
        self,
        query: torch.Tensor,
        img_feats: Sequence[torch.Tensor],
        projection_mat: torch.Tensor | None = None,
        image_wh: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self_out = self.self_attn(query, query, query, need_weights=False)[0]
        query = self.norm1(query + self_out)
        cross_out = self.cross_attn(query, img_feats, projection_mat=projection_mat, image_wh=image_wh)
        query = self.norm2(query + cross_out)
        ffn_out = self.ffn(query)
        query = self.norm3(query + ffn_out)
        return query


class SparseBox3DRefinementLite(nn.Module):
    """Predicts class logits and refined 3D boxes from decoder states."""

    def __init__(self, embed_dims: int, num_classes: int, box_code_size: int) -> None:
        super().__init__()
        self.box_code_size = int(box_code_size)
        self.cls_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, num_classes),
        )
        self.reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, box_code_size),
        )

    def forward(self, query: torch.Tensor, anchors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls_scores = self.cls_branch(query)
        delta = self.reg_branch(query)
        boxes = anchors + delta
        if self.box_code_size >= 6:
            size = torch.exp(boxes[..., 3:6].clamp(min=-4.0, max=4.0))
            boxes = torch.cat([boxes[..., :3], size, boxes[..., 6:]], dim=-1)
        return cls_scores, boxes
