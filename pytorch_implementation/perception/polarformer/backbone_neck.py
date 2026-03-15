"""Lightweight image backbone/FPN + polar projection neck."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .config import BackboneNeckConfig, PolarNeckConfig
from .utils import (
    SinePositionalEncoding2D,
    stack_camera_matrices,
    validate_polarformer_img_metas,
)


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleBackbone(nn.Module):
    """Compact CNN backbone with four stages."""

    def __init__(self, stage_channels: Sequence[int]) -> None:
        super().__init__()
        if len(stage_channels) != 4:
            raise ValueError("SimpleBackbone expects exactly four stage channels.")
        c1, c2, c3, c4 = stage_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.stages = nn.ModuleList(
            [
                _ConvBlock(c1, c1, stride=1),
                _ConvBlock(c1, c2, stride=2),
                _ConvBlock(c2, c3, stride=2),
                _ConvBlock(c3, c4, stride=2),
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


class SimpleFPN(nn.Module):
    """Minimal FPN with optional extra levels."""

    def __init__(self, in_channels: Sequence[int], out_channels: int, num_outs: int) -> None:
        super().__init__()
        if num_outs < len(in_channels):
            raise ValueError("num_outs must be >= number of input feature levels.")
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels]
        )
        self.output_convs = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels]
        )
        self.extra_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
                for _ in range(num_outs - len(in_channels))
            ]
        )

    def forward(self, inputs: Iterable[torch.Tensor]) -> list[torch.Tensor]:
        feats = list(inputs)
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, feats)]
        for idx in range(len(laterals) - 1, 0, -1):
            laterals[idx - 1] = laterals[idx - 1] + F.interpolate(
                laterals[idx], size=laterals[idx - 1].shape[-2:], mode="nearest"
            )
        outs = [conv(x) for conv, x in zip(self.output_convs, laterals)]
        if self.extra_convs:
            cur = outs[-1]
            for conv in self.extra_convs:
                cur = conv(cur)
                outs.append(cur)
        return outs


class PolarRayCrossAttentionLite(nn.Module):
    """Cross-attention from polar rays (queries) to image columns (keys)."""

    def __init__(self, embed_dims: int, num_heads: int) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=0.0)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims * 2, embed_dims),
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

    def forward(self, img_columns: torch.Tensor, polar_rays: torch.Tensor) -> torch.Tensor:
        query2 = self.cross_attn(polar_rays, img_columns, img_columns, need_weights=False)[0]
        polar_rays = self.norm1(polar_rays + query2)
        polar_rays = self.norm2(polar_rays + self.ffn(polar_rays))
        return polar_rays


class BackboneNeck(nn.Module):
    """Backbone/FPN plus PolarFormer-style polar projection neck."""

    def __init__(self, backbone_cfg: BackboneNeckConfig, polar_cfg: PolarNeckConfig) -> None:
        super().__init__()
        self.backbone_cfg = backbone_cfg
        self.polar_cfg = polar_cfg
        self.backbone = SimpleBackbone(backbone_cfg.stage_channels)
        selected_in_channels = [backbone_cfg.stage_channels[idx] for idx in backbone_cfg.out_indices]
        self.fpn = SimpleFPN(
            in_channels=selected_in_channels,
            out_channels=backbone_cfg.out_channels,
            num_outs=backbone_cfg.num_outs,
        )
        self.polar_projectors = nn.ModuleList(
            [
                PolarRayCrossAttentionLite(backbone_cfg.out_channels, polar_cfg.num_heads)
                for _ in range(polar_cfg.num_levels)
            ]
        )
        self.polar_out_convs = nn.ModuleList(
            [
                nn.Conv2d(backbone_cfg.out_channels, backbone_cfg.out_channels, kernel_size=3, padding=1)
                for _ in range(polar_cfg.num_levels)
            ]
        )
        self.position_encoding = SinePositionalEncoding2D(backbone_cfg.out_channels // 2, normalize=True)
        self.ray_geometry_proj = nn.Sequential(
            nn.Linear(6, backbone_cfg.out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_cfg.out_channels, backbone_cfg.out_channels),
        )

    def _target_size(self, level_idx: int) -> tuple[int, int]:
        azimuth, radius, _ = self.polar_cfg.output_size
        if not self.polar_cfg.use_different_res:
            return int(radius), int(azimuth)
        scale = 2**level_idx
        return max(1, int(radius // scale)), max(1, int(azimuth // scale))

    @staticmethod
    def _meta_hw_tensor(
        img_metas: list[dict[str, Any]],
        *,
        batch_size: int,
        num_cams: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        img_hw = torch.zeros((batch_size, num_cams, 2), device=device, dtype=dtype)
        for batch_idx, meta in enumerate(img_metas):
            for cam_idx in range(num_cams):
                img_shape = meta["img_shape"][cam_idx]
                img_hw[batch_idx, cam_idx, 0] = float(img_shape[0])
                img_hw[batch_idx, cam_idx, 1] = float(img_shape[1])
        return img_hw

    def _build_polar_ray_geometry(
        self,
        *,
        radius_bins: int,
        width: int,
        cam_intrinsic: torch.Tensor,
        cam2lidar: torch.Tensor,
        img_hw: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Build [R, B*W, C] geometry-conditioned polar ray embeddings."""

        if radius_bins <= 0 or width <= 0:
            raise ValueError(f"radius_bins and width must be positive, got {(radius_bins, width)}")

        radius_start, radius_end, _ = self.polar_cfg.radius_range
        radius_span = max(float(radius_end - radius_start), 1e-5)
        if radius_bins == 1:
            radius = torch.full((1,), 0.5 * (radius_start + radius_end), device=device, dtype=dtype)
        else:
            radius_step = radius_span / float(radius_bins)
            radius = torch.arange(radius_bins, device=device, dtype=dtype) * radius_step + (
                radius_start + 0.5 * radius_step
            )
        radius_norm = ((radius - radius_start) / radius_span).clamp(0.0, 1.0)

        feature_x = torch.arange(width, device=device, dtype=dtype) + 0.5
        img_w = img_hw[:, 1].clamp(min=1.0)
        stride_x = img_w / max(width, 1)
        pixel_x = feature_x.unsqueeze(0) * stride_x.unsqueeze(-1)  # [B, W]

        fx = cam_intrinsic[:, 0, 0].clamp(min=1e-5)
        cx = cam_intrinsic[:, 0, 2]
        column_theta = torch.atan2(pixel_x - cx.unsqueeze(-1), fx.unsqueeze(-1))  # [B, W]

        camera_forward_xy = cam2lidar[:, :2, 2]
        camera_yaw = torch.atan2(camera_forward_xy[:, 0], camera_forward_xy[:, 1]).unsqueeze(-1)  # [B, 1]
        ray_theta = column_theta + camera_yaw

        batch_size = cam_intrinsic.shape[0]
        sin_theta = ray_theta.sin().unsqueeze(1).expand(batch_size, radius_bins, width)
        cos_theta = ray_theta.cos().unsqueeze(1).expand(batch_size, radius_bins, width)
        radius_feat = radius_norm.view(1, radius_bins, 1).expand(batch_size, radius_bins, width)
        camera_xy = cam2lidar[:, :2, 3]
        camera_xy_norm = torch.tanh(camera_xy / max(abs(float(radius_end)), 1.0))
        camera_x = camera_xy_norm[:, 0].view(batch_size, 1, 1).expand(batch_size, radius_bins, width)
        camera_y = camera_xy_norm[:, 1].view(batch_size, 1, 1).expand(batch_size, radius_bins, width)
        col_angle = (column_theta / torch.pi).unsqueeze(1).expand(batch_size, radius_bins, width)

        ray_geom = torch.stack((sin_theta, cos_theta, radius_feat, camera_x, camera_y, col_angle), dim=-1)
        ray_geom = self.ray_geometry_proj(ray_geom.reshape(batch_size * radius_bins * width, 6))
        return ray_geom.view(batch_size, radius_bins, width, -1).permute(1, 0, 2, 3).reshape(
            radius_bins, batch_size * width, -1
        )

    def _project_single_camera(
        self,
        feat: torch.Tensor,
        *,
        projector: PolarRayCrossAttentionLite,
        radius_bins: int,
        cam_intrinsic: torch.Tensor,
        cam2lidar: torch.Tensor,
        img_hw: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, channels, height, width = feat.shape
        image_mask = torch.zeros((batch_size, height, width), device=feat.device, dtype=torch.bool)
        image_pos = self.position_encoding(image_mask)
        polar_mask = torch.zeros((batch_size, radius_bins, width), device=feat.device, dtype=torch.bool)
        polar_pos = self.position_encoding(polar_mask)
        ray_geometry = self._build_polar_ray_geometry(
            radius_bins=radius_bins,
            width=width,
            cam_intrinsic=cam_intrinsic,
            cam2lidar=cam2lidar,
            img_hw=img_hw,
            dtype=feat.dtype,
            device=feat.device,
        )

        img_columns = (feat + image_pos).permute(2, 0, 3, 1).reshape(height, batch_size * width, channels)
        polar_rays = polar_pos.permute(2, 0, 3, 1).reshape(radius_bins, batch_size * width, channels)
        polar_rays = polar_rays + ray_geometry
        polar = projector(img_columns, polar_rays)
        return polar.reshape(radius_bins, batch_size, width, channels).permute(1, 3, 0, 2).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        *,
        batch_size: int,
        num_cams: int,
        img_metas: list[dict[str, Any]],
        strict_img_meta: bool = True,
        require_geometry: bool = True,
    ) -> list[torch.Tensor]:
        validate_polarformer_img_metas(
            img_metas,
            batch_size=batch_size,
            num_cams=num_cams,
            strict_img_meta=strict_img_meta,
            require_geometry=require_geometry,
        )
        if require_geometry:
            cam_intrinsic = stack_camera_matrices(
                img_metas,
                field_name="cam_intrinsic",
                num_cams=num_cams,
                device=x.device,
                dtype=x.dtype,
                expected_shape=(3, 3),
            )
            cam2lidar = stack_camera_matrices(
                img_metas,
                field_name="cam2lidar",
                num_cams=num_cams,
                device=x.device,
                dtype=x.dtype,
                expected_shape=(4, 4),
            )
        else:
            identity3 = torch.eye(3, device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
            identity4 = torch.eye(4, device=x.device, dtype=x.dtype).view(1, 1, 4, 4)
            cam_intrinsic = identity3.expand(batch_size, num_cams, -1, -1).contiguous()
            cam2lidar = identity4.expand(batch_size, num_cams, -1, -1).contiguous()
        img_hw = self._meta_hw_tensor(
            img_metas,
            batch_size=batch_size,
            num_cams=num_cams,
            device=x.device,
            dtype=x.dtype,
        )

        stages = self.backbone(x)
        selected = [stages[idx] for idx in self.backbone_cfg.out_indices]
        fpn_feats = self.fpn(selected)
        fpn_feats = [
            feat.view(batch_size, num_cams, feat.shape[1], feat.shape[2], feat.shape[3]) for feat in fpn_feats
        ]

        outputs: list[torch.Tensor] = []
        for level_idx in range(self.polar_cfg.num_levels):
            level_feat = fpn_feats[level_idx]
            target_r, target_a = self._target_size(level_idx)
            fused = torch.zeros(
                (batch_size, self.backbone_cfg.out_channels, target_r, target_a),
                device=level_feat.device,
                dtype=level_feat.dtype,
            )
            for cam_idx in range(num_cams):
                polar_feat = self._project_single_camera(
                    level_feat[:, cam_idx],
                    projector=self.polar_projectors[level_idx],
                    radius_bins=target_r,
                    cam_intrinsic=cam_intrinsic[:, cam_idx],
                    cam2lidar=cam2lidar[:, cam_idx],
                    img_hw=img_hw[:, cam_idx],
                )
                polar_feat = F.interpolate(
                    polar_feat,
                    size=(target_r, target_a),
                    mode="bilinear",
                    align_corners=False,
                )
                fused = fused + polar_feat
            fused = fused / max(1, num_cams)
            outputs.append(self.polar_out_convs[level_idx](fused))
        return outputs

