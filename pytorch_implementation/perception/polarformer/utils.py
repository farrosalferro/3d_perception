"""Math and positional-encoding helpers for PolarFormer-lite."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch

from ...common.meta.validators import (
    CameraMetaProfile,
    stack_camera_matrices as _shared_stack_camera_matrices,
    validate_camera_img_metas,
)
from ...common.utils.numerics import inverse_sigmoid as _shared_inverse_sigmoid
from ...common.utils.positional_encoding import SinePositionalEncoding2D


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Numerically stable inverse sigmoid."""
    return _shared_inverse_sigmoid(x, eps=eps)


_POLARFORMER_BASE_KEYS: tuple[str, ...] = ("img_shape",)


def validate_polarformer_img_metas(
    img_metas: list[dict[str, Any]],
    *,
    batch_size: int | None = None,
    num_cams: int | None = None,
    strict_img_meta: bool = True,
    require_geometry: bool = True,
) -> None:
    """Validate the subset of `img_metas` contract used by PolarFormer forward."""
    required_keys = list(_POLARFORMER_BASE_KEYS)
    matrix_exact_shapes: dict[str, tuple[int, int]] = {}
    matrix_min_shapes: dict[str, tuple[int, int]] = {}
    if require_geometry:
        required_keys.extend(["lidar2img", "cam_intrinsic", "cam2lidar"])
        matrix_exact_shapes = {"lidar2img": (4, 4), "cam2lidar": (4, 4)}
        matrix_min_shapes = {"cam_intrinsic": (3, 3)}
    profile = CameraMetaProfile(
        required_keys=tuple(required_keys),
        require_pad_shape=strict_img_meta,
        enforce_pad_greater_equal_img=strict_img_meta,
        matrix_exact_shapes=matrix_exact_shapes,
        matrix_min_shapes=matrix_min_shapes,
    )
    validate_camera_img_metas(
        img_metas,
        profile=profile,
        batch_size=batch_size,
        num_cams=num_cams,
    )


def stack_camera_matrices(
    img_metas: list[dict[str, Any]],
    *,
    field_name: str,
    num_cams: int,
    device: torch.device,
    dtype: torch.dtype,
    expected_shape: tuple[int, int],
) -> torch.Tensor:
    """Build [B, Ncam, H, W] camera matrix tensor from `img_metas`."""
    return _shared_stack_camera_matrices(
        img_metas,
        field_name=field_name,
        num_cams=num_cams,
        device=device,
        dtype=dtype,
        expected_shape=expected_shape,
    )


def denormalize_bbox(normalized_bboxes: torch.Tensor, pc_range: Sequence[float]) -> torch.Tensor:
    """Restore PolarFormer box format from center/polar residual parameterization."""

    del pc_range  # kept for API parity with upstream helper signature

    delta_rot_sine = normalized_bboxes[..., 6:7]
    delta_rot_cosine = normalized_bboxes[..., 7:8]
    delta_rot = torch.atan2(delta_rot_sine, delta_rot_cosine)

    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    theta_center = torch.atan2(cx, cy)
    rot = theta_center + delta_rot
    rot = torch.remainder(rot + math.pi, 2 * math.pi) - math.pi

    w = normalized_bboxes[..., 2:3].exp()
    l = normalized_bboxes[..., 3:4].exp()
    h = normalized_bboxes[..., 5:6].exp()

    if normalized_bboxes.size(-1) > 8:
        v_theta = normalized_bboxes[..., 8:9]
        v_r = normalized_bboxes[..., 9:10]
        v_abs = torch.sqrt(v_theta.square() + v_r.square())
        delta_vel = torch.atan2(v_theta, v_r)
        v_dir = delta_vel + theta_center
        vx = v_abs * torch.sin(v_dir)
        vy = v_abs * torch.cos(v_dir)
        return torch.cat((cx, cy, cz, w, l, h, rot, vx, vy), dim=-1)
    return torch.cat((cx, cy, cz, w, l, h, rot), dim=-1)



