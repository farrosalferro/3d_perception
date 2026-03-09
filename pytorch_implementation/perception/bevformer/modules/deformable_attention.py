"""Pure PyTorch deformable-attention implementations for BEVFormer."""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as F
from torch import nn


def ms_deform_attn_torch(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """A simple multi-scale deformable attention kernel in pure PyTorch.

    Args:
        value: [B, S, num_heads, head_dim]
        spatial_shapes: [num_levels, 2] where each row is (H, W)
        sampling_locations: [B, Q, num_heads, num_levels, num_points, 2], normalized [0, 1]
        attention_weights: [B, Q, num_heads, num_levels, num_points]
    """

    batch_size, _, num_heads, head_dim = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape

    split_sizes = (spatial_shapes[:, 0] * spatial_shapes[:, 1]).tolist()
    value_per_level = value.split(split_sizes, dim=1)
    output = value.new_zeros(batch_size, num_queries, num_heads, head_dim)

    for level_index, (h_lvl, w_lvl) in enumerate(spatial_shapes.tolist()):
        value_level = value_per_level[level_index]
        value_level = value_level.reshape(batch_size, h_lvl, w_lvl, num_heads, head_dim)
        value_level = value_level.permute(0, 3, 4, 1, 2).reshape(batch_size * num_heads, head_dim, h_lvl, w_lvl)

        grid_level = sampling_locations[:, :, :, level_index, :, :]
        grid_level = grid_level.permute(0, 2, 1, 3, 4).reshape(batch_size * num_heads, num_queries, num_points, 2)
        grid_level = grid_level * 2.0 - 1.0

        sampled = F.grid_sample(
            value_level,
            grid_level,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )  # [B * H, head_dim, Q, P]
        sampled = sampled.view(batch_size, num_heads, head_dim, num_queries, num_points)
        sampled = sampled.permute(0, 3, 1, 4, 2)  # [B, Q, H, P, head_dim]

        weights_level = attention_weights[:, :, :, level_index, :].unsqueeze(-1)
        output = output + (sampled * weights_level).sum(dim=3)

    return output.reshape(batch_size, num_queries, num_heads * head_dim)


def _is_power_of_2(value: int) -> bool:
    return (value & (value - 1) == 0) and value != 0


class CustomMSDeformableAttentionLite(nn.Module):
    """Decoder deformable attention without custom CUDA ops."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        dropout: float = 0.1,
        batch_first: bool = False,
    ) -> None:
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, got {embed_dims}/{num_heads}")
        dim_per_head = embed_dims // num_heads
        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "Per-head dim is not a power of 2. This is okay for pure PyTorch,"
                " but can be slower than optimized kernels.",
                stacklevel=2,
            )

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)

        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
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
        grid_init = grid_init.repeat(1, self.num_levels, self.num_points, 1)
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
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if reference_points is None or spatial_shapes is None:
            raise ValueError("reference_points and spatial_shapes are required.")

        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        batch_size, num_query, _ = query.shape
        _, num_value, _ = value.shape
        expected_num_value = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item())
        if expected_num_value != num_value:
            raise ValueError(f"num_value mismatch: expected {expected_num_value}, got {num_value}")

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(batch_size, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            batch_size, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            batch_size, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(dim=-1).view(
            batch_size, num_query, self.num_heads, self.num_levels, self.num_points
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
            raise ValueError(f"Last dim of reference_points must be 2 or 4, got {reference_points.shape[-1]}")

        output = ms_deform_attn_torch(value, spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return self.dropout(output) + identity


class MSDeformableAttention3DLite(nn.Module):
    """Spatial deformable attention used inside camera cross-attention."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 8,
        batch_first: bool = True,
    ) -> None:
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, got {embed_dims}/{num_heads}")
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.batch_first = batch_first

        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = (grid_init / grid_init.abs().max(dim=-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 2
        )
        grid_init = grid_init.repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.reshape(-1)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)

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
        del key, identity, level_start_index, kwargs
        if value is None:
            value = query
        if query_pos is not None:
            query = query + query_pos
        if reference_points is None or spatial_shapes is None:
            raise ValueError("reference_points and spatial_shapes are required.")

        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        batch_size, num_query, _ = query.shape
        _, num_value, _ = value.shape
        expected_num_value = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item())
        if expected_num_value != num_value:
            raise ValueError(f"num_value mismatch: expected {expected_num_value}, got {num_value}")

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(batch_size, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            batch_size, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            batch_size, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(dim=-1).view(
            batch_size, num_query, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] != 2:
            raise ValueError(f"Expected reference_points[..., 2], got {reference_points.shape[-1]}")

        # Each BEV query has multiple projected anchors along height.
        offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1)
        num_z_anchors = reference_points.shape[2]
        reference_points = reference_points[:, :, None, None, None, :, :]
        sampling_offsets = sampling_offsets / offset_normalizer[None, None, None, :, None, :]

        _, _, _, _, num_all_points, xy_dims = sampling_offsets.shape
        if num_all_points % num_z_anchors != 0:
            raise ValueError("num_points must be divisible by num_z_anchors in this simplified implementation.")

        sampling_offsets = sampling_offsets.view(
            batch_size,
            num_query,
            self.num_heads,
            self.num_levels,
            num_all_points // num_z_anchors,
            num_z_anchors,
            xy_dims,
        )
        sampling_locations = reference_points + sampling_offsets
        sampling_locations = sampling_locations.view(
            batch_size, num_query, self.num_heads, self.num_levels, num_all_points, xy_dims
        )

        output = ms_deform_attn_torch(value, spatial_shapes, sampling_locations, attention_weights)
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return output
