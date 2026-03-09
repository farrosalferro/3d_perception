"""Geometry helpers for BEV reference points and camera projection."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
import torch


def get_reference_points_3d(
    bev_h: int,
    bev_w: int,
    z_extent: float,
    num_points_in_pillar: int,
    bs: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Reference points for spatial cross-attention.

    Returns shape: [bs, num_points_in_pillar, bev_h * bev_w, 3].
    """

    zs = (
        torch.linspace(0.5, z_extent - 0.5, num_points_in_pillar, dtype=dtype, device=device)
        .view(-1, 1, 1)
        .expand(num_points_in_pillar, bev_h, bev_w)
        / z_extent
    )
    xs = (
        torch.linspace(0.5, bev_w - 0.5, bev_w, dtype=dtype, device=device)
        .view(1, 1, bev_w)
        .expand(num_points_in_pillar, bev_h, bev_w)
        / bev_w
    )
    ys = (
        torch.linspace(0.5, bev_h - 0.5, bev_h, dtype=dtype, device=device)
        .view(1, bev_h, 1)
        .expand(num_points_in_pillar, bev_h, bev_w)
        / bev_h
    )
    ref_3d = torch.stack((xs, ys, zs), dim=-1)
    ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
    return ref_3d[None].repeat(bs, 1, 1, 1)


def get_reference_points_2d(
    bev_h: int,
    bev_w: int,
    bs: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Reference points for temporal self-attention.

    Returns shape: [bs, bev_h * bev_w, 1, 2].
    """

    ref_y, ref_x = torch.meshgrid(
        torch.linspace(0.5, bev_h - 0.5, bev_h, dtype=dtype, device=device),
        torch.linspace(0.5, bev_w - 0.5, bev_w, dtype=dtype, device=device),
        indexing="ij",
    )
    ref_y = ref_y.reshape(-1)[None] / bev_h
    ref_x = ref_x.reshape(-1)[None] / bev_w
    ref_2d = torch.stack((ref_x, ref_y), dim=-1)
    return ref_2d.repeat(bs, 1, 1).unsqueeze(2)


def _infer_image_hw(img_shape: object) -> Tuple[float, float]:
    """Infer image height/width from common metadata formats."""

    if isinstance(img_shape, torch.Tensor):
        values = img_shape.tolist()
        if len(values) >= 2:
            return float(values[0]), float(values[1])
    if isinstance(img_shape, np.ndarray):
        values = img_shape.tolist()
        if len(values) >= 2:
            return float(values[0]), float(values[1])
    if isinstance(img_shape, (list, tuple)):
        if len(img_shape) == 0:
            return 1.0, 1.0
        first = img_shape[0]
        if isinstance(first, (list, tuple, np.ndarray, torch.Tensor)):
            return _infer_image_hw(first)
        if len(img_shape) >= 2:
            return float(img_shape[0]), float(img_shape[1])
    raise ValueError(f"Unsupported img_shape format: {type(img_shape)}")


def point_sampling(
    reference_points: torch.Tensor,
    pc_range: Sequence[float],
    img_metas: Iterable[dict],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project BEV 3D reference points into each camera image.

    Args:
        reference_points: [B, D, num_query, 3], normalized in [0, 1].
        pc_range: point cloud range [x_min, y_min, z_min, x_max, y_max, z_max].
        img_metas: metadata list containing `lidar2img` and `img_shape`.

    Returns:
        reference_points_cam: [num_cam, B, num_query, D, 2]
        bev_mask: [num_cam, B, num_query, D]
    """

    img_metas = list(img_metas)
    lidar2img = []
    for meta in img_metas:
        lidar2img.append(meta["lidar2img"])
    lidar2img = np.asarray(lidar2img)
    lidar2img = reference_points.new_tensor(lidar2img)  # [B, num_cam, 4, 4]

    reference_points = reference_points.clone()
    reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
    reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), dim=-1)

    # [B, D, num_query, 4] -> [D, B, num_query, 4]
    reference_points = reference_points.permute(1, 0, 2, 3)
    depth_bins, batch_size, num_query = reference_points.size()[:3]
    num_cam = lidar2img.size(1)

    reference_points = reference_points.view(depth_bins, batch_size, 1, num_query, 4)
    reference_points = reference_points.repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

    lidar2img = lidar2img.view(1, batch_size, num_cam, 1, 4, 4).repeat(depth_bins, 1, 1, num_query, 1, 1)
    reference_points_cam = torch.matmul(lidar2img.float(), reference_points.float()).squeeze(-1)

    eps = 1e-5
    bev_mask = reference_points_cam[..., 2:3] > eps
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps
    )

    img_h, img_w = _infer_image_hw(img_metas[0]["img_shape"])
    reference_points_cam[..., 0] /= img_w
    reference_points_cam[..., 1] /= img_h

    bev_mask = (
        bev_mask
        & (reference_points_cam[..., 1:2] > 0.0)
        & (reference_points_cam[..., 1:2] < 1.0)
        & (reference_points_cam[..., 0:1] > 0.0)
        & (reference_points_cam[..., 0:1] < 1.0)
    )
    bev_mask = torch.nan_to_num(bev_mask)

    reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
    bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)
    return reference_points_cam.to(dtype=reference_points.dtype), bev_mask
