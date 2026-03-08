"""Detection decoder for pure-PyTorch BEVFormer."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from ..utils.math import inverse_sigmoid
from .deformable_attention import CustomMSDeformableAttentionLite


class _DecoderLayerLite(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        ffn_dims: int,
        num_heads: int,
        cross_num_levels: int = 1,
        cross_num_points: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=False)
        self.cross_attn = CustomMSDeformableAttentionLite(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=cross_num_levels,
            num_points=cross_num_points,
            dropout=dropout,
            batch_first=False,
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, ffn_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dims, embed_dims),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)
        self.dropout1 = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        *,
        key: Optional[torch.Tensor],
        value: torch.Tensor,
        query_pos: Optional[torch.Tensor],
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = query + query_pos if query_pos is not None else query
        self_attn_out, _ = self.self_attn(q, q, query)
        query = self.norm1(query + self.dropout1(self_attn_out))

        query = self.cross_attn(
            query,
            key=key,
            value=value,
            identity=query,
            query_pos=query_pos,
            key_padding_mask=key_padding_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        query = self.norm2(query)
        query = self.norm3(query + self.ffn(query))
        return query


class DetectionTransformerDecoderLite(nn.Module):
    """Forward-only decoder aligned with BEVFormer behavior."""

    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        ffn_dims: int,
        num_heads: int,
        return_intermediate: bool = True,
        cross_num_levels: int = 1,
        cross_num_points: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.return_intermediate = return_intermediate
        self.layers = nn.ModuleList(
            [
                _DecoderLayerLite(
                    embed_dims=embed_dims,
                    ffn_dims=ffn_dims,
                    num_heads=num_heads,
                    cross_num_levels=cross_num_levels,
                    cross_num_points=cross_num_points,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        query: torch.Tensor,
        *,
        key: torch.Tensor | None,
        value: torch.Tensor,
        reference_points: torch.Tensor,
        reg_branches: nn.ModuleList | None = None,
        key_padding_mask: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = query
        intermediate = []
        intermediate_reference_points = []

        for layer_index, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(2)
            output = layer(
                output,
                key=key,
                value=value,
                query_pos=query_pos,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask,
            )
            output_batch_first = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[layer_index](output_batch_first)
                if reference_points.shape[-1] != 3:
                    raise ValueError("reference_points must have last dim=3.")
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
                reference_points = new_reference_points.sigmoid().detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        return output, reference_points
