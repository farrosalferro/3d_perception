"""BEVFormer encoder stack in pure PyTorch."""

from __future__ import annotations

from typing import List

import torch
from torch import nn

from ..utils.geometry import get_reference_points_2d, get_reference_points_3d, point_sampling
from .spatial_cross_attention import SpatialCrossAttentionLite
from .temporal_self_attention import TemporalSelfAttentionLite


class _FFNBlock(nn.Module):
    def __init__(self, embed_dims: int, ffn_dims: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dims, ffn_dims)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dims, embed_dims)
        self.output_dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0.0)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.output_dropout(self.linear2(self.dropout(self.act(self.linear1(x)))))


class BEVFormerLayerLite(nn.Module):
    """One encoder layer: temporal self-attn -> spatial cross-attn -> FFN."""

    def __init__(
        self,
        embed_dims: int = 256,
        ffn_dims: int = 512,
        dropout: float = 0.1,
        num_cams: int = 6,
        num_heads: int = 8,
        temporal_num_levels: int = 1,
        temporal_num_points: int = 4,
        spatial_num_levels: int = 4,
        spatial_num_points: int = 8,
    ) -> None:
        super().__init__()
        self.temporal_attn = TemporalSelfAttentionLite(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=temporal_num_levels,
            num_points=temporal_num_points,
            batch_first=True,
            dropout=dropout,
        )
        self.spatial_attn = SpatialCrossAttentionLite(
            embed_dims=embed_dims,
            num_cams=num_cams,
            dropout=dropout,
            num_levels=spatial_num_levels,
            num_points=spatial_num_points,
        )
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(3)])
        self.ffn = _FFNBlock(embed_dims=embed_dims, ffn_dims=ffn_dims, dropout=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        bev_pos: torch.Tensor,
        ref_2d: torch.Tensor,
        ref_3d: torch.Tensor,
        bev_h: int,
        bev_w: int,
        reference_points_cam: torch.Tensor,
        bev_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        prev_bev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self.temporal_attn(
            query,
            prev_bev,
            prev_bev,
            identity=query,
            query_pos=bev_pos,
            reference_points=ref_2d,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
        )
        query = self.norms[0](query)

        query = self.spatial_attn(
            query,
            key,
            value,
            residual=query,
            query_pos=None,
            reference_points=ref_3d,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)
        return query


class BEVFormerEncoderLite(nn.Module):
    """Pure-PyTorch BEVFormer encoder."""

    def __init__(
        self,
        num_layers: int,
        pc_range: List[float] | tuple[float, ...],
        num_points_in_pillar: int = 4,
        return_intermediate: bool = False,
        *,
        embed_dims: int = 256,
        ffn_dims: int = 512,
        dropout: float = 0.1,
        num_cams: int = 6,
        num_heads: int = 8,
        temporal_num_levels: int = 1,
        temporal_num_points: int = 4,
        spatial_num_levels: int = 4,
        spatial_num_points: int = 8,
    ) -> None:
        super().__init__()
        self.pc_range = list(pc_range)
        self.num_points_in_pillar = num_points_in_pillar
        self.return_intermediate = return_intermediate
        self.layers = nn.ModuleList(
            [
                BEVFormerLayerLite(
                    embed_dims=embed_dims,
                    ffn_dims=ffn_dims,
                    dropout=dropout,
                    num_cams=num_cams,
                    num_heads=num_heads,
                    temporal_num_levels=temporal_num_levels,
                    temporal_num_points=temporal_num_points,
                    spatial_num_levels=spatial_num_levels,
                    spatial_num_points=spatial_num_points,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        bev_query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        bev_h: int,
        bev_w: int,
        bev_pos: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        img_metas: list[dict],
        prev_bev: torch.Tensor | None = None,
        shift: torch.Tensor | float = 0.0,
    ) -> torch.Tensor:
        output = bev_query
        intermediate = []

        ref_3d = get_reference_points_3d(
            bev_h,
            bev_w,
            self.pc_range[5] - self.pc_range[2],
            self.num_points_in_pillar,
            bs=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype,
        )
        ref_2d = get_reference_points_2d(
            bev_h,
            bev_w,
            bs=bev_query.size(1),
            device=bev_query.device,
            dtype=bev_query.dtype,
        )
        reference_points_cam, bev_mask = point_sampling(ref_3d, self.pc_range, img_metas)

        shift_ref_2d = ref_2d.clone()
        if isinstance(shift, torch.Tensor):
            shift_ref_2d = shift_ref_2d + shift[:, None, None, :]

        bev_query = bev_query.permute(1, 0, 2)  # [bs, HW, C]
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape

        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = torch.stack([prev_bev, bev_query], dim=1).reshape(bs * 2, len_bev, -1)
            hybrid_ref_2d = torch.stack([shift_ref_2d, ref_2d], dim=1).reshape(
                bs * 2, len_bev, num_bev_level, 2
            )
        else:
            hybrid_ref_2d = torch.stack([ref_2d, ref_2d], dim=1).reshape(bs * 2, len_bev, num_bev_level, 2)

        for layer in self.layers:
            output = layer(
                bev_query,
                key,
                value,
                bev_pos=bev_pos,
                ref_2d=hybrid_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
            )
            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        return output
