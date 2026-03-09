"""Perception transformer used by BEVFormer."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

from .decoder import DetectionTransformerDecoderLite
from .encoder import BEVFormerEncoderLite


def _to_float_tensor(values: Sequence[float], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(values, device=device, dtype=dtype)


class PerceptionTransformerLite(nn.Module):
    """Forward-only PerceptionTransformer in pure PyTorch."""

    def __init__(
        self,
        *,
        encoder: BEVFormerEncoderLite,
        decoder: DetectionTransformerDecoderLite,
        embed_dims: int = 256,
        num_feature_levels: int = 4,
        num_cams: int = 6,
        rotate_prev_bev: bool = True,
        use_shift: bool = True,
        use_can_bus: bool = True,
        can_bus_norm: bool = True,
        use_cams_embeds: bool = True,
        can_bus_dims: int = 18,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.use_cams_embeds = use_cams_embeds

        self.level_embeds = nn.Parameter(torch.empty(num_feature_levels, embed_dims))
        self.cams_embeds = nn.Parameter(torch.empty(num_cams, embed_dims))
        self.reference_points = nn.Linear(embed_dims, 3)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(can_bus_dims, embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims // 2, embed_dims),
            nn.ReLU(inplace=True),
        )
        if can_bus_norm:
            self.can_bus_mlp.add_module("norm", nn.LayerNorm(embed_dims))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.level_embeds)
        nn.init.normal_(self.cams_embeds)
        nn.init.xavier_uniform_(self.reference_points.weight)
        nn.init.constant_(self.reference_points.bias, 0.0)
        for module in self.can_bus_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def get_bev_features(
        self,
        mlvl_feats: list[torch.Tensor],
        bev_queries: torch.Tensor,
        bev_h: int,
        bev_w: int,
        *,
        grid_length: tuple[float, float],
        bev_pos: torch.Tensor,
        img_metas: list[dict],
        prev_bev: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        device = bev_queries.device
        dtype = bev_queries.dtype
        delta_x = torch.stack([_to_float_tensor(meta["can_bus"], device=device, dtype=dtype)[0] for meta in img_metas])
        delta_y = torch.stack([_to_float_tensor(meta["can_bus"], device=device, dtype=dtype)[1] for meta in img_metas])
        ego_angle = torch.stack(
            [_to_float_tensor(meta["can_bus"], device=device, dtype=dtype)[-2] / math.pi * 180.0 for meta in img_metas]
        )
        translation_length = torch.sqrt(delta_x**2 + delta_y**2)
        translation_angle = torch.atan2(delta_y, delta_x) / math.pi * 180.0
        bev_angle = ego_angle - translation_angle
        grid_length_y, grid_length_x = grid_length

        shift_y = translation_length * torch.cos(bev_angle / 180.0 * math.pi) / grid_length_y / bev_h
        shift_x = translation_length * torch.sin(bev_angle / 180.0 * math.pi) / grid_length_x / bev_w
        if not self.use_shift:
            shift_x = torch.zeros_like(shift_x)
            shift_y = torch.zeros_like(shift_y)
        shift = torch.stack([shift_x, shift_y], dim=-1)

        # The original code rotates previous BEV according to ego motion.
        # For this pure-PyTorch learning implementation we keep it as a no-op.
        if prev_bev is not None and self.rotate_prev_bev:
            prev_bev = prev_bev

        can_bus = torch.stack([_to_float_tensor(meta["can_bus"], device=device, dtype=dtype) for meta in img_metas], dim=0)
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        if self.use_can_bus:
            bev_queries = bev_queries + can_bus

        feat_flatten = []
        spatial_shapes = []
        for level_index, feat in enumerate(mlvl_feats):
            bs, num_cam, channels, h_lvl, w_lvl = feat.shape
            spatial_shapes.append((h_lvl, w_lvl))
            feat = feat.flatten(3).permute(1, 0, 3, 2)  # [Ncam, B, HW, C]
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, level_index : level_index + 1, :].to(feat.dtype)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, dim=2)
        spatial_shapes_tensor = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat(
            (
                spatial_shapes_tensor.new_zeros((1,)),
                spatial_shapes_tensor.prod(1).cumsum(0)[:-1],
            )
        )
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # [Ncam, sum(HW), B, C]

        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes_tensor,
            level_start_index=level_start_index,
            img_metas=img_metas,
            prev_bev=prev_bev,
            shift=shift,
        )
        return bev_embed

    def forward(
        self,
        mlvl_feats: list[torch.Tensor],
        bev_queries: torch.Tensor,
        object_query_embed: torch.Tensor,
        bev_h: int,
        bev_w: int,
        *,
        grid_length: tuple[float, float],
        bev_pos: torch.Tensor,
        img_metas: list[dict],
        reg_branches: nn.ModuleList | None = None,
        cls_branches: nn.ModuleList | None = None,
        prev_bev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del cls_branches
        bev_embed = self.get_bev_features(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            img_metas=img_metas,
            prev_bev=prev_bev,
        )  # [B, HW, C]

        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos).sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)  # [Q, B, C]
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)  # [HW, B, C]

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
        )
        return bev_embed, inter_states, init_reference_out, inter_references
