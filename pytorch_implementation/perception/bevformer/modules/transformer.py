"""Perception transformer used by BEVFormer."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F
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
        rotate_center: tuple[float, float] = (100.0, 100.0),
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
        self.can_bus_dims = can_bus_dims
        if len(rotate_center) != 2:
            raise ValueError("rotate_center must contain two values [x, y].")
        self.rotate_center = (float(rotate_center[0]), float(rotate_center[1]))

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

    def _validate_img_metas(self, img_metas: list[dict[str, Any]], *, batch_size: int) -> None:
        if not isinstance(img_metas, list):
            raise TypeError(f"img_metas must be a list of dicts, got {type(img_metas)}")
        if len(img_metas) != batch_size:
            raise ValueError(f"img_metas length mismatch: expected {batch_size}, got {len(img_metas)}")

        required_keys = ("can_bus", "lidar2img", "img_shape")
        for sample_idx, meta in enumerate(img_metas):
            if not isinstance(meta, dict):
                raise TypeError(f"img_metas[{sample_idx}] must be a dict, got {type(meta)}")
            missing_keys = [key for key in required_keys if key not in meta]
            if missing_keys:
                raise KeyError(f"img_metas[{sample_idx}] missing required keys: {missing_keys}")

            try:
                lidar2img = torch.as_tensor(meta["lidar2img"])
            except Exception as exc:  # pragma: no cover - defensive branch
                raise ValueError(f"img_metas[{sample_idx}]['lidar2img'] is not tensor-like.") from exc
            if lidar2img.ndim != 3 or lidar2img.shape[0] != self.num_cams or lidar2img.shape[-2:] != (4, 4):
                raise ValueError(
                    f"img_metas[{sample_idx}]['lidar2img'] must have shape "
                    f"[{self.num_cams}, 4, 4], got {tuple(lidar2img.shape)}"
                )

            img_shape = meta["img_shape"]
            if isinstance(img_shape, (list, tuple)) and len(img_shape) == 0:
                raise ValueError(f"img_metas[{sample_idx}]['img_shape'] cannot be empty.")

    def _stack_can_bus(
        self,
        img_metas: list[dict[str, Any]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        can_bus_list = []
        for sample_idx, meta in enumerate(img_metas):
            can_bus = _to_float_tensor(meta["can_bus"], device=device, dtype=dtype).flatten()
            if can_bus.numel() != self.can_bus_dims:
                raise ValueError(
                    f"img_metas[{sample_idx}]['can_bus'] must contain {self.can_bus_dims} values, "
                    f"got {can_bus.numel()}"
                )
            can_bus_list.append(can_bus)
        return torch.stack(can_bus_list, dim=0)

    def _normalize_prev_bev_layout(
        self,
        prev_bev: torch.Tensor,
        *,
        bev_h: int,
        bev_w: int,
        batch_size: int,
    ) -> torch.Tensor:
        if prev_bev.dim() != 3:
            raise ValueError(f"prev_bev must be a 3D tensor, got shape {tuple(prev_bev.shape)}")
        bev_tokens = bev_h * bev_w
        if prev_bev.shape[0] == bev_tokens and prev_bev.shape[1] == batch_size:
            return prev_bev
        if prev_bev.shape[0] == batch_size and prev_bev.shape[1] == bev_tokens:
            return prev_bev.permute(1, 0, 2)
        raise ValueError(
            "prev_bev must have shape [bev_h*bev_w, B, C] or [B, bev_h*bev_w, C], "
            f"got {tuple(prev_bev.shape)} with bev_h={bev_h}, bev_w={bev_w}, B={batch_size}"
        )

    @staticmethod
    def _rotate_feature_map(
        feature_map: torch.Tensor,
        *,
        angle_degrees: float,
        center: tuple[float, float],
    ) -> torch.Tensor:
        """Rotate `[C, H, W]` feature map by angle around pixel center."""
        if feature_map.dim() != 3:
            raise ValueError(f"feature_map must have shape [C, H, W], got {tuple(feature_map.shape)}")
        if angle_degrees == 0.0:
            return feature_map

        _, feat_h, feat_w = feature_map.shape
        center_x, center_y = center
        radians = angle_degrees * math.pi / 180.0
        cos_theta = math.cos(radians)
        sin_theta = math.sin(radians)

        # Grid samples input coordinates for each output pixel. This reproduces
        # image-like rotation behavior without torchvision/mmcv dependencies.
        y_coords = torch.arange(feat_h, device=feature_map.device, dtype=feature_map.dtype)
        x_coords = torch.arange(feat_w, device=feature_map.device, dtype=feature_map.dtype)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")

        x_rel = grid_x - center_x
        y_rel = grid_y - center_y
        src_x = cos_theta * x_rel + sin_theta * y_rel + center_x
        src_y = -sin_theta * x_rel + cos_theta * y_rel + center_y

        src_x = ((src_x + 0.5) / feat_w) * 2.0 - 1.0
        src_y = ((src_y + 0.5) / feat_h) * 2.0 - 1.0
        sampling_grid = torch.stack((src_x, src_y), dim=-1).unsqueeze(0)

        input_dtype = feature_map.dtype
        sampled_input = feature_map.unsqueeze(0)
        if sampled_input.device.type == "cpu" and input_dtype in {torch.float16, torch.bfloat16}:
            sampled_input = sampled_input.float()
            sampling_grid = sampling_grid.float()
        rotated = F.grid_sample(
            sampled_input,
            sampling_grid.to(dtype=sampled_input.dtype),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return rotated.squeeze(0).to(dtype=input_dtype)

    def _rotate_prev_bev(
        self,
        prev_bev: torch.Tensor,
        *,
        rotation_angles: torch.Tensor,
        bev_h: int,
        bev_w: int,
    ) -> torch.Tensor:
        """Rotate each sample in `prev_bev` with per-batch ego yaw delta."""
        rotated_prev_bev = prev_bev.clone()
        for batch_idx in range(prev_bev.shape[1]):
            rotation_angle = float(rotation_angles[batch_idx].item())
            if rotation_angle == 0.0:
                continue
            prev_bev_per_batch = prev_bev[:, batch_idx].reshape(bev_h, bev_w, -1).permute(2, 0, 1)
            prev_bev_per_batch = self._rotate_feature_map(
                prev_bev_per_batch,
                angle_degrees=rotation_angle,
                center=self.rotate_center,
            )
            rotated_prev_bev[:, batch_idx] = prev_bev_per_batch.permute(1, 2, 0).reshape(bev_h * bev_w, -1)
        return rotated_prev_bev

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
        if len(mlvl_feats) != self.num_feature_levels:
            raise ValueError(
                f"Expected {self.num_feature_levels} feature levels, got {len(mlvl_feats)}."
            )
        self._validate_img_metas(img_metas, batch_size=bs)

        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        device = bev_queries.device
        dtype = bev_queries.dtype
        can_bus = self._stack_can_bus(img_metas, device=device, dtype=dtype)
        delta_x = can_bus[:, 0]
        delta_y = can_bus[:, 1]
        ego_angle = can_bus[:, -2] / math.pi * 180.0
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

        if prev_bev is not None:
            prev_bev = self._normalize_prev_bev_layout(prev_bev, bev_h=bev_h, bev_w=bev_w, batch_size=bs)
            if self.rotate_prev_bev:
                prev_bev = self._rotate_prev_bev(
                    prev_bev,
                    rotation_angles=can_bus[:, -1],
                    bev_h=bev_h,
                    bev_w=bev_w,
                )

        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        if self.use_can_bus:
            bev_queries = bev_queries + can_bus

        feat_flatten = []
        spatial_shapes = []
        for level_index, feat in enumerate(mlvl_feats):
            bs, num_cam, channels, h_lvl, w_lvl = feat.shape
            if num_cam != self.num_cams:
                raise ValueError(
                    f"Feature level {level_index} has {num_cam} cameras, expected {self.num_cams}."
                )
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
