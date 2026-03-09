"""Spatial cross-attention for BEVFormer."""

from __future__ import annotations

import torch
from torch import nn

from .deformable_attention import MSDeformableAttention3DLite


class SpatialCrossAttentionLite(nn.Module):
    """Camera-aware cross-attention over BEV queries."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_cams: int = 6,
        dropout: float = 0.1,
        num_levels: int = 4,
        num_points: int = 8,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MSDeformableAttention3DLite(
            embed_dims=embed_dims,
            num_levels=num_levels,
            num_points=num_points,
            batch_first=True,
        )
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        residual: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        reference_points: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        reference_points_cam: torch.Tensor | None = None,
        bev_mask: torch.Tensor | None = None,
        level_start_index: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del key_padding_mask, reference_points, kwargs
        if reference_points_cam is None or bev_mask is None:
            raise ValueError("reference_points_cam and bev_mask are required.")
        if spatial_shapes is None or level_start_index is None:
            raise ValueError("spatial_shapes and level_start_index are required.")
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        else:
            inp_residual = residual
            slots = torch.zeros_like(query)

        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()
        num_depth = reference_points_cam.size(3)

        indexes = []
        for mask_per_img in bev_mask:
            # Keep this behavior close to the original implementation.
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
        max_len = max(len(each) for each in indexes) if indexes else 0

        if max_len == 0:
            slots = self.output_proj(slots)
            return self.dropout(slots) + inp_residual

        queries_rebatch = query.new_zeros((bs, self.num_cams, max_len, self.embed_dims))
        reference_points_rebatch = reference_points_cam.new_zeros((bs, self.num_cams, max_len, num_depth, 2))

        for batch_idx in range(bs):
            for cam_idx, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = indexes[cam_idx]
                valid_len = len(index_query_per_img)
                if valid_len == 0:
                    continue
                queries_rebatch[batch_idx, cam_idx, :valid_len] = query[batch_idx, index_query_per_img]
                reference_points_rebatch[batch_idx, cam_idx, :valid_len] = reference_points_per_img[
                    batch_idx, index_query_per_img
                ]

        _, feature_len, _, _ = key.shape  # [num_cam, L, bs, C]
        key = key.permute(2, 0, 1, 3).reshape(bs * self.num_cams, feature_len, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(bs * self.num_cams, feature_len, self.embed_dims)

        queries = self.deformable_attention(
            query=queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims),
            key=key,
            value=value,
            reference_points=reference_points_rebatch.view(bs * self.num_cams, max_len, num_depth, 2),
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        ).view(bs, self.num_cams, max_len, self.embed_dims)

        for batch_idx in range(bs):
            for cam_idx, index_query_per_img in enumerate(indexes):
                valid_len = len(index_query_per_img)
                if valid_len == 0:
                    continue
                slots[batch_idx, index_query_per_img] += queries[batch_idx, cam_idx, :valid_len]

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)
        return self.dropout(slots) + inp_residual
