"""Temporal self-attention used in BEVFormer encoder."""

from __future__ import annotations

import math

import torch
from torch import nn

from .deformable_attention import ms_deform_attn_torch


class TemporalSelfAttentionLite(nn.Module):
    """A pure-PyTorch variant of BEVFormer temporal self-attention."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 1,
        num_points: int = 4,
        num_bev_queue: int = 2,
        dropout: float = 0.1,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, got {embed_dims}/{num_heads}")
        if num_bev_queue != 2:
            raise ValueError("This forward-only implementation currently supports num_bev_queue=2 only.")
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)

        in_dims = embed_dims * num_bev_queue
        self.sampling_offsets = nn.Linear(
            in_dims, num_bev_queue * num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            in_dims, num_bev_queue * num_heads * num_levels * num_points
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = (grid_init / grid_init.abs().max(dim=-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 2
        )
        grid_init = grid_init.repeat(1, self.num_levels * self.num_bev_queue, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.reshape(-1)

        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        identity: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        reference_points: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        level_start_index: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del key, level_start_index, kwargs
        if not self.batch_first:
            raise ValueError("TemporalSelfAttentionLite expects batch_first=True.")
        if reference_points is None or spatial_shapes is None:
            raise ValueError("reference_points and spatial_shapes are required.")

        if value is None:
            bs, len_bev, channels = query.shape
            value = torch.stack([query, query], dim=1).reshape(bs * 2, len_bev, channels)
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.shape
        _, num_value, _ = value.shape
        expected_num_value = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item())
        if expected_num_value != num_value:
            raise ValueError(f"num_value mismatch: expected {expected_num_value}, got {num_value}")

        query = torch.cat([value[:bs], query], dim=-1)
        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)

        value = value.reshape(bs * self.num_bev_queue, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs,
            num_query,
            self.num_heads,
            self.num_bev_queue,
            self.num_levels,
            self.num_points,
            2,
        )
        attention_weights = self.attention_weights(query).view(
            bs,
            num_query,
            self.num_heads,
            self.num_bev_queue,
            self.num_levels * self.num_points,
        )
        attention_weights = attention_weights.softmax(dim=-1)
        attention_weights = attention_weights.view(
            bs,
            num_query,
            self.num_heads,
            self.num_bev_queue,
            self.num_levels,
            self.num_points,
        )

        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5).reshape(
            bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points
        )
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6).reshape(
            bs * self.num_bev_queue,
            num_query,
            self.num_heads,
            self.num_levels,
            self.num_points,
            2,
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"reference_points last dim must be 2 or 4, got {reference_points.shape[-1]}")

        output = ms_deform_attn_torch(value, spatial_shapes, sampling_locations, attention_weights)
        output = output.permute(1, 2, 0).view(num_query, self.embed_dims, bs, self.num_bev_queue).mean(dim=-1)
        output = output.permute(2, 0, 1)
        output = self.output_proj(output)
        return self.dropout(output) + identity
