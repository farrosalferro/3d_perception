"""Backward projection refinement for FB-BEV."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .config import FBBEVForwardConfig
from .depth_aware_attention import DepthAwareAttentionLite


class BackwardProjectionLite(nn.Module):
    """Refine BEV features with depth-aware camera-to-BEV fusion."""

    def __init__(self, cfg: FBBEVForwardConfig) -> None:
        super().__init__()
        self.depth_attention = DepthAwareAttentionLite(cfg.embed_dims)
        self.post = nn.Sequential(
            nn.Conv2d(cfg.embed_dims, cfg.embed_dims, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.embed_dims = int(cfg.embed_dims)
        self.depth_bins = int(cfg.depth_bins)
        self.bev_h = int(cfg.bev_h)
        self.bev_w = int(cfg.bev_w)
        self.bev_z = int(cfg.bev_z)
        self.pc_range = tuple(float(v) for v in cfg.pc_range)
        self.depth_start = float(cfg.depth_range[0])
        self.depth_end = float(cfg.depth_range[1])
        self.num_z_anchors = max(1, min(self.bev_z, 4))

    def _fallback_projected_context(
        self,
        bev_2d: torch.Tensor,
        context: torch.Tensor,
        depth_prob: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context_bev = context.mean(dim=1)
        depth_weight = depth_prob.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        context_bev = F.interpolate(context_bev, size=bev_2d.shape[-2:], mode="bilinear", align_corners=False)
        depth_weight = F.interpolate(depth_weight, size=bev_2d.shape[-2:], mode="bilinear", align_corners=False)
        return context_bev, depth_weight.clamp(min=0.0, max=1.0)

    def _build_reference_points(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        x_step = (self.pc_range[3] - self.pc_range[0]) / max(self.bev_w, 1)
        y_step = (self.pc_range[4] - self.pc_range[1]) / max(self.bev_h, 1)
        z_step = (self.pc_range[5] - self.pc_range[2]) / max(self.num_z_anchors, 1)
        x = torch.linspace(
            self.pc_range[0] + x_step * 0.5,
            self.pc_range[3] - x_step * 0.5,
            self.bev_w,
            device=device,
            dtype=dtype,
        )
        y = torch.linspace(
            self.pc_range[1] + y_step * 0.5,
            self.pc_range[4] - y_step * 0.5,
            self.bev_h,
            device=device,
            dtype=dtype,
        )
        z = torch.linspace(
            self.pc_range[2] + z_step * 0.5,
            self.pc_range[5] - z_step * 0.5,
            self.num_z_anchors,
            device=device,
            dtype=dtype,
        )
        grid_y, grid_x, grid_z = torch.meshgrid(y, x, z, indexing="ij")
        return torch.stack((grid_x, grid_y, grid_z), dim=-1)  # [Hbev, Wbev, Zanchor, 3]

    def _read_img_hw(
        self,
        img_metas: list[dict[str, Any]] | None,
        *,
        batch_size: int,
        num_cams: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        img_hw = torch.ones(batch_size, num_cams, 2, device=device, dtype=dtype)
        if not img_metas:
            return img_hw
        for batch_idx, meta in enumerate(img_metas):
            if batch_idx >= batch_size:
                break
            shapes = meta.get("img_shape")
            if not isinstance(shapes, (list, tuple)):
                continue
            for cam_idx in range(min(num_cams, len(shapes))):
                shape = shapes[cam_idx]
                if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                    img_hw[batch_idx, cam_idx, 0] = float(shape[0])
                    img_hw[batch_idx, cam_idx, 1] = float(shape[1])
        return img_hw

    def _project_with_lidar2img(
        self,
        reference_points: torch.Tensor,
        lidar2img: torch.Tensor,
        img_hw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_cams = lidar2img.shape[:2]
        h_bev, w_bev, z_anchor, _ = reference_points.shape
        num_query = h_bev * w_bev * z_anchor

        points = reference_points.reshape(num_query, 3)
        points_homo = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)
        points_homo = points_homo.view(1, 1, num_query, 4, 1).expand(batch_size, num_cams, -1, -1, -1)
        proj = lidar2img.view(batch_size, num_cams, 1, 4, 4).matmul(points_homo).squeeze(-1)
        eps = 1e-5
        depth = proj[..., 2:3]
        xy = proj[..., 0:2] / torch.maximum(depth, torch.full_like(depth, eps))
        img_h = img_hw[..., 0].unsqueeze(-1).clamp(min=1.0)
        img_w = img_hw[..., 1].unsqueeze(-1).clamp(min=1.0)
        x_norm = xy[..., 0] / img_w
        y_norm = xy[..., 1] / img_h
        ref_cam = torch.stack((x_norm, y_norm), dim=-1)
        mask = (
            (depth[..., 0] > eps)
            & (ref_cam[..., 0] > eps)
            & (ref_cam[..., 0] < (1.0 - eps))
            & (ref_cam[..., 1] > eps)
            & (ref_cam[..., 1] < (1.0 - eps))
        )
        ref_cam = ref_cam.view(batch_size, num_cams, h_bev, w_bev, z_anchor, 2)
        mask = mask.view(batch_size, num_cams, h_bev, w_bev, z_anchor)
        depth = depth[..., 0].view(batch_size, num_cams, h_bev, w_bev, z_anchor)
        return ref_cam, mask, depth

    def _project_with_cam_params(
        self,
        reference_points: torch.Tensor,
        cam_params: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        img_hw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rots, trans, intrins, post_rots, post_trans, bda = cam_params
        batch_size, num_cams, _ = trans.shape
        h_bev, w_bev, z_anchor, _ = reference_points.shape
        points = reference_points.view(1, 1, h_bev, w_bev, z_anchor, 3).expand(batch_size, num_cams, -1, -1, -1, -1)
        points = torch.linalg.inv(bda).view(batch_size, 1, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        points = points - trans.view(batch_size, num_cams, 1, 1, 1, 3)
        combine = torch.linalg.inv(rots.matmul(torch.linalg.inv(intrins)))
        points_cam = combine.view(batch_size, num_cams, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        eps = 1e-5
        points_cam = torch.cat(
            [
                points_cam[..., 0:2]
                / torch.maximum(points_cam[..., 2:3], torch.full_like(points_cam[..., 2:3], eps)),
                points_cam[..., 2:3],
            ],
            dim=-1,
        )
        points_cam = post_rots.view(batch_size, num_cams, 1, 1, 1, 3, 3).matmul(points_cam.unsqueeze(-1)).squeeze(-1)
        points_cam = points_cam + post_trans.view(batch_size, num_cams, 1, 1, 1, 3)

        img_h = img_hw[..., 0].view(batch_size, num_cams, 1, 1, 1).clamp(min=1.0)
        img_w = img_hw[..., 1].view(batch_size, num_cams, 1, 1, 1).clamp(min=1.0)
        ref_cam = torch.stack((points_cam[..., 0] / img_w, points_cam[..., 1] / img_h), dim=-1)
        depth = points_cam[..., 2]
        mask = (
            (depth > eps)
            & (ref_cam[..., 0] > eps)
            & (ref_cam[..., 0] < (1.0 - eps))
            & (ref_cam[..., 1] > eps)
            & (ref_cam[..., 1] < (1.0 - eps))
        )
        return ref_cam, mask, depth

    def _sample_projected_context(
        self,
        context: torch.Tensor,
        depth_prob: torch.Tensor,
        reference_points_cam: torch.Tensor,
        reference_mask: torch.Tensor,
        query_depth: torch.Tensor,
        *,
        bev_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        batch_size, num_cams, channels, feat_h, feat_w = context.shape
        _, _, h_bev, w_bev, z_anchor, _ = reference_points_cam.shape

        grid = reference_points_cam.permute(0, 1, 4, 2, 3, 5).reshape(batch_size * num_cams * z_anchor, h_bev, w_bev, 2)
        grid = grid * 2.0 - 1.0

        context_expanded = (
            context.unsqueeze(2)
            .expand(batch_size, num_cams, z_anchor, channels, feat_h, feat_w)
            .reshape(batch_size * num_cams * z_anchor, channels, feat_h, feat_w)
        )
        sampled_context = F.grid_sample(
            context_expanded,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled_context = sampled_context.view(batch_size, num_cams, z_anchor, channels, h_bev, w_bev)

        depth_expanded = (
            depth_prob.unsqueeze(2)
            .expand(batch_size, num_cams, z_anchor, self.depth_bins, feat_h, feat_w)
            .reshape(batch_size * num_cams * z_anchor, self.depth_bins, feat_h, feat_w)
        )
        sampled_depth = F.grid_sample(
            depth_expanded,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled_depth = sampled_depth.view(batch_size, num_cams, z_anchor, self.depth_bins, h_bev, w_bev)

        depth_span = max(self.depth_end - self.depth_start, 1e-6)
        depth_index = ((query_depth - self.depth_start) / depth_span) * float(max(self.depth_bins - 1, 1))
        depth_index = depth_index.round().clamp(min=0, max=max(self.depth_bins - 1, 0)).to(torch.long)
        depth_index = depth_index.permute(0, 1, 4, 2, 3).unsqueeze(3)
        sampled_prob = torch.gather(sampled_depth, dim=3, index=depth_index).squeeze(3)

        valid = reference_mask.permute(0, 1, 4, 2, 3)
        if bev_mask is not None:
            if bev_mask.shape != (batch_size, h_bev, w_bev):
                raise ValueError(
                    f"bev_mask must be [{batch_size}, {h_bev}, {w_bev}], got {tuple(bev_mask.shape)}"
                )
            valid = valid & bev_mask[:, None, None, :, :].to(torch.bool)

        sampled_prob = sampled_prob * valid.to(dtype=sampled_prob.dtype)
        weighted_context = sampled_context * sampled_prob.unsqueeze(3)
        context_sum = weighted_context.sum(dim=(1, 2))  # [B, C, H, W]
        weight_sum = sampled_prob.sum(dim=(1, 2))  # [B, H, W]
        context_bev = context_sum / weight_sum.unsqueeze(1).clamp(min=1e-6)
        depth_weight = (weight_sum / float(max(num_cams * z_anchor, 1))).unsqueeze(1).clamp(min=0.0, max=1.0)
        return context_bev, depth_weight, bool(valid.any())

    def _project_context_with_geometry(
        self,
        context: torch.Tensor,
        depth_prob: torch.Tensor,
        *,
        cam_params: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
        lidar2img: torch.Tensor | None = None,
        img_metas: list[dict[str, Any]] | None = None,
        bev_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        batch_size, num_cams = context.shape[:2]
        reference_points = self._build_reference_points(device=context.device, dtype=context.dtype)
        img_hw = self._read_img_hw(
            img_metas,
            batch_size=batch_size,
            num_cams=num_cams,
            device=context.device,
            dtype=context.dtype,
        )
        if lidar2img is not None:
            if lidar2img.shape != (batch_size, num_cams, 4, 4):
                raise ValueError(
                    f"lidar2img must be [{batch_size}, {num_cams}, 4, 4], got {tuple(lidar2img.shape)}"
                )
            ref_cam, ref_mask, query_depth = self._project_with_lidar2img(reference_points, lidar2img, img_hw)
        elif cam_params is not None:
            ref_cam, ref_mask, query_depth = self._project_with_cam_params(reference_points, cam_params, img_hw)
        else:
            raise ValueError("Geometry projection requires either lidar2img or cam_params.")
        return self._sample_projected_context(
            context,
            depth_prob,
            ref_cam,
            ref_mask,
            query_depth,
            bev_mask=bev_mask,
        )

    def forward(
        self,
        bev_volume: torch.Tensor,
        context: torch.Tensor,
        depth_prob: torch.Tensor,
        *,
        cam_params: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
        lidar2img: torch.Tensor | None = None,
        img_metas: list[dict[str, Any]] | None = None,
        bev_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Args:
        bev_volume: [B, C, Hbev, Wbev, Zbev]
        context: [B, Ncam, C, Hf, Wf]
        depth_prob: [B, Ncam, D, Hf, Wf]
        Returns:
            refined_bev: [B, C, Hbev, Wbev]
        """

        if bev_volume.dim() != 5:
            raise ValueError(f"Expected bev_volume [B, C, H, W, Z], got {tuple(bev_volume.shape)}")
        if context.dim() != 5 or depth_prob.dim() != 5:
            raise ValueError("Expected context/depth_prob with shape [B, Ncam, *, H, W].")
        if context.shape[0] != bev_volume.shape[0] or depth_prob.shape[0] != bev_volume.shape[0]:
            raise ValueError("bev_volume/context/depth_prob batch dimensions must match.")

        bev_2d = bev_volume.mean(dim=-1)
        context_bev, depth_weight = self._fallback_projected_context(bev_2d, context, depth_prob)
        if cam_params is not None or lidar2img is not None:
            try:
                projected_context, projected_depth_weight, has_valid = self._project_context_with_geometry(
                    context,
                    depth_prob,
                    cam_params=cam_params,
                    lidar2img=lidar2img,
                    img_metas=img_metas,
                    bev_mask=bev_mask,
                )
            except Exception:
                has_valid = False
            else:
                if has_valid:
                    context_bev = projected_context
                    depth_weight = projected_depth_weight

        refined = self.depth_attention(bev_2d, context_bev, depth_weight)
        return self.post(refined)
