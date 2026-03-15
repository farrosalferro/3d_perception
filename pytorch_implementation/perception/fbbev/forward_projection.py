"""Pure-PyTorch forward projection from image-view features to BEV volume."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .config import FBBEVForwardConfig


class ForwardProjectionLite(nn.Module):
    """Lift-splat style projection with optional camera-geometry parity path."""

    def __init__(self, cfg: FBBEVForwardConfig) -> None:
        super().__init__()
        self.bev_h = int(cfg.bev_h)
        self.bev_w = int(cfg.bev_w)
        self.bev_z = int(cfg.bev_z)
        self.embed_dims = int(cfg.embed_dims)
        self.depth_bins = int(cfg.depth_bins)
        self.depth_start = float(cfg.depth_range[0])
        self.depth_end = float(cfg.depth_range[1])
        self.pc_range = tuple(float(v) for v in cfg.pc_range)

        x_interval = (self.pc_range[3] - self.pc_range[0]) / max(self.bev_w, 1)
        y_interval = (self.pc_range[4] - self.pc_range[1]) / max(self.bev_h, 1)
        z_interval = (self.pc_range[5] - self.pc_range[2]) / max(self.bev_z, 1)
        self.register_buffer(
            "grid_lower_bound",
            torch.tensor([self.pc_range[0], self.pc_range[1], self.pc_range[2]], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "grid_interval",
            torch.tensor([x_interval, y_interval, z_interval], dtype=torch.float32),
            persistent=False,
        )

    def _validate_inputs(self, context: torch.Tensor, depth_prob: torch.Tensor) -> None:
        if context.dim() != 5 or depth_prob.dim() != 5:
            raise ValueError("Expected context/depth with shape [B, Ncam, *, H, W].")
        if context.shape[0] != depth_prob.shape[0] or context.shape[1] != depth_prob.shape[1]:
            raise ValueError("Context/depth batch and camera dimensions must match.")
        if context.shape[-2:] != depth_prob.shape[-2:]:
            raise ValueError("Context/depth feature resolutions must match.")
        if depth_prob.shape[2] != self.depth_bins:
            raise ValueError(f"Expected depth bins={self.depth_bins}, got {depth_prob.shape[2]}.")
        if context.shape[2] != self.embed_dims:
            raise ValueError(f"Expected context channels={self.embed_dims}, got {context.shape[2]}.")
        if not torch.isfinite(context).all() or not torch.isfinite(depth_prob).all():
            raise ValueError("Forward projection inputs must be finite.")

    def _infer_image_hw(
        self,
        img_metas: list[dict[str, Any]] | None,
        feat_h: int,
        feat_w: int,
    ) -> tuple[float, float]:
        if not img_metas:
            return float(feat_h), float(feat_w)
        shape_entry = img_metas[0].get("img_shape")
        if not isinstance(shape_entry, (list, tuple)) or len(shape_entry) == 0:
            return float(feat_h), float(feat_w)
        first_shape = shape_entry[0]
        if isinstance(first_shape, (list, tuple)) and len(first_shape) >= 2:
            return float(first_shape[0]), float(first_shape[1])
        return float(feat_h), float(feat_w)

    def _forward_depth_weighted_resize(self, context: torch.Tensor, depth_prob: torch.Tensor) -> torch.Tensor:
        """Fallback path used when camera geometry is unavailable."""

        volume = context.unsqueeze(2) * depth_prob.unsqueeze(3)  # [B, Ncam, D, C, H, W]
        volume = volume.mean(dim=1)  # [B, D, C, H, W]
        volume = volume.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, D, H, W]
        volume = F.interpolate(
            volume,
            size=(self.bev_z, self.bev_h, self.bev_w),
            mode="trilinear",
            align_corners=False,
        )
        return volume.permute(0, 1, 3, 4, 2).contiguous()

    def _validate_cam_params(
        self,
        cam_params: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        batch_size: int,
        num_cams: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(cam_params) != 6:
            raise ValueError("cam_params must contain (rots, trans, intrins, post_rots, post_trans, bda).")
        rots, trans, intrins, post_rots, post_trans, bda = cam_params
        rots = torch.as_tensor(rots, device=device, dtype=dtype)
        trans = torch.as_tensor(trans, device=device, dtype=dtype)
        intrins = torch.as_tensor(intrins, device=device, dtype=dtype)
        post_rots = torch.as_tensor(post_rots, device=device, dtype=dtype)
        post_trans = torch.as_tensor(post_trans, device=device, dtype=dtype)
        bda = torch.as_tensor(bda, device=device, dtype=dtype)
        if bda.dim() == 3 and bda.shape[-2:] == (4, 4):
            bda = bda[:, :3, :3]
        if rots.shape != (batch_size, num_cams, 3, 3):
            raise ValueError(f"rots must be [{batch_size}, {num_cams}, 3, 3], got {tuple(rots.shape)}")
        if trans.shape != (batch_size, num_cams, 3):
            raise ValueError(f"trans must be [{batch_size}, {num_cams}, 3], got {tuple(trans.shape)}")
        if intrins.shape != (batch_size, num_cams, 3, 3):
            raise ValueError(f"intrins must be [{batch_size}, {num_cams}, 3, 3], got {tuple(intrins.shape)}")
        if post_rots.shape != (batch_size, num_cams, 3, 3):
            raise ValueError(f"post_rots must be [{batch_size}, {num_cams}, 3, 3], got {tuple(post_rots.shape)}")
        if post_trans.shape != (batch_size, num_cams, 3):
            raise ValueError(f"post_trans must be [{batch_size}, {num_cams}, 3], got {tuple(post_trans.shape)}")
        if bda.shape != (batch_size, 3, 3):
            raise ValueError(f"bda must be [{batch_size}, 3, 3], got {tuple(bda.shape)}")
        for name, tensor in {
            "rots": rots,
            "trans": trans,
            "intrins": intrins,
            "post_rots": post_rots,
            "post_trans": post_trans,
            "bda": bda,
        }.items():
            if not torch.isfinite(tensor).all():
                raise ValueError(f"cam_params '{name}' must be finite.")
        return rots, trans, intrins, post_rots, post_trans, bda

    def _create_frustum(
        self,
        feat_h: int,
        feat_w: int,
        *,
        img_h: float,
        img_w: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        depth = torch.linspace(self.depth_start, self.depth_end, self.depth_bins, device=device, dtype=dtype)
        depth = depth.view(self.depth_bins, 1, 1).expand(self.depth_bins, feat_h, feat_w)
        x = torch.linspace(0.0, max(img_w - 1.0, 0.0), feat_w, device=device, dtype=dtype)
        y = torch.linspace(0.0, max(img_h - 1.0, 0.0), feat_h, device=device, dtype=dtype)
        x = x.view(1, 1, feat_w).expand(self.depth_bins, feat_h, feat_w)
        y = y.view(1, feat_h, 1).expand(self.depth_bins, feat_h, feat_w)
        return torch.stack((x, y, depth), dim=-1)  # [D, Hf, Wf, 3]

    def _get_lidar_coor(
        self,
        cam_params: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        frustum: torch.Tensor,
    ) -> torch.Tensor:
        rots, trans, intrins, post_rots, post_trans, bda = cam_params
        batch_size, num_cams, _ = trans.shape
        points = frustum.view(1, 1, *frustum.shape).expand(batch_size, num_cams, -1, -1, -1, -1)
        points = points - post_trans.view(batch_size, num_cams, 1, 1, 1, 3)
        points = torch.linalg.inv(post_rots).view(batch_size, num_cams, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        points = points.squeeze(-1)
        points = torch.cat((points[..., :2] * points[..., 2:3], points[..., 2:3]), dim=-1)
        combine = rots.matmul(torch.linalg.inv(intrins))
        points = combine.view(batch_size, num_cams, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        points = points + trans.view(batch_size, num_cams, 1, 1, 1, 3)
        points = bda.view(batch_size, 1, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        return points  # [B, N, D, Hf, Wf, 3]

    def _scatter_to_bev(self, points: torch.Tensor, weighted_feat: torch.Tensor) -> tuple[torch.Tensor, bool]:
        feat = weighted_feat.permute(0, 1, 2, 4, 5, 3).contiguous()  # [B, N, D, Hf, Wf, C]
        batch_size, _, _, _, _, channels = feat.shape

        lower_bound = self.grid_lower_bound.to(device=points.device, dtype=points.dtype)
        interval = self.grid_interval.to(device=points.device, dtype=points.dtype)
        coords = ((points - lower_bound) / interval).long()
        x_idx = coords[..., 0]
        y_idx = coords[..., 1]
        z_idx = coords[..., 2]
        valid = (
            (x_idx >= 0)
            & (x_idx < self.bev_w)
            & (y_idx >= 0)
            & (y_idx < self.bev_h)
            & (z_idx >= 0)
            & (z_idx < self.bev_z)
        )
        linear = z_idx * (self.bev_h * self.bev_w) + y_idx * self.bev_w + x_idx
        bev_flat = feat.new_zeros(batch_size, channels, self.bev_h * self.bev_w * self.bev_z)

        has_points = False
        for batch_idx in range(batch_size):
            mask = valid[batch_idx].reshape(-1)
            if not bool(mask.any()):
                continue
            has_points = True
            dst_idx = linear[batch_idx].reshape(-1)[mask].to(torch.long)
            src_feat = feat[batch_idx].reshape(-1, channels)[mask]  # [P, C]
            bev_flat[batch_idx].scatter_add_(1, dst_idx.unsqueeze(0).expand(channels, -1), src_feat.transpose(0, 1))

        bev = bev_flat.view(batch_size, channels, self.bev_z, self.bev_h, self.bev_w)
        bev = bev.permute(0, 1, 3, 4, 2).contiguous()  # [B, C, Hbev, Wbev, Zbev]
        return bev, has_points

    def forward(
        self,
        context: torch.Tensor,
        depth_prob: torch.Tensor,
        *,
        cam_params: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
        img_metas: list[dict[str, Any]] | None = None,
    ) -> torch.Tensor:
        """Build a BEV volume from depth-weighted camera features.

        Args:
            context: [B, Ncam, C, Hf, Wf]
            depth_prob: [B, Ncam, D, Hf, Wf]
            cam_params: optional (rots, trans, intrins, post_rots, post_trans, bda).
            img_metas: optional metadata used for image-shape scaling.
        Returns:
            bev_volume: [B, C, Hbev, Wbev, Zbev]
        """

        self._validate_inputs(context, depth_prob)
        fallback = self._forward_depth_weighted_resize(context, depth_prob)
        if cam_params is None:
            return fallback

        batch_size, num_cams, _, feat_h, feat_w = context.shape
        try:
            cam_params_valid = self._validate_cam_params(
                cam_params,
                batch_size=batch_size,
                num_cams=num_cams,
                device=context.device,
                dtype=context.dtype,
            )
        except Exception:
            return fallback

        img_h, img_w = self._infer_image_hw(img_metas, feat_h, feat_w)
        frustum = self._create_frustum(
            feat_h,
            feat_w,
            img_h=img_h,
            img_w=img_w,
            device=context.device,
            dtype=context.dtype,
        )
        lidar_points = self._get_lidar_coor(cam_params_valid, frustum)
        weighted_feat = context.unsqueeze(2) * depth_prob.unsqueeze(3)  # [B, N, D, C, Hf, Wf]
        bev, has_points = self._scatter_to_bev(lidar_points, weighted_feat)
        if not has_points or not torch.isfinite(bev).all():
            return fallback
        return bev
