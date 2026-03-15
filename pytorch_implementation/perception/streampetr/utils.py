"""Math, metadata, memory, and positional-encoding helpers for StreamPETR."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch

from ...common.meta.validators import (
    CameraMetaProfile,
    build_img2lidars_from_metas,
    validate_camera_img_metas,
)
from ...common.postprocess.gather import gather_dim1_topk
from ...common.utils.numerics import inverse_sigmoid as _shared_inverse_sigmoid
from ...common.utils.positional_encoding import SinePositionalEncoding2D


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Numerically stable inverse sigmoid."""
    return _shared_inverse_sigmoid(x, eps=eps)


def pos2posemb3d(pos: torch.Tensor, num_pos_feats: int = 128, temperature: int = 10000) -> torch.Tensor:
    """Sinusoidal 3D embedding for normalized reference points."""

    scale = 2.0 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    return torch.cat((pos_y, pos_x, pos_z), dim=-1)


def pos2posemb1d(pos: torch.Tensor, num_pos_feats: int = 128, temperature: int = 10000) -> torch.Tensor:
    """Sinusoidal 1D embedding, typically for temporal timestamps."""

    scale = 2.0 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
    pos_x = pos[..., None] / dim_t
    return torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)


def denormalize_bbox(normalized_bboxes: torch.Tensor, pc_range: Sequence[float]) -> torch.Tensor:
    """Mirror StreamPETR/MMDet3D decode-time bbox restoration."""

    del pc_range  # Kept for signature parity with upstream helper.
    rot_sine = normalized_bboxes[..., 6:7]
    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 2:3]
    w = normalized_bboxes[..., 3:4].exp()
    l = normalized_bboxes[..., 4:5].exp()
    h = normalized_bboxes[..., 5:6].exp()

    if normalized_bboxes.size(-1) > 8:
        vx = normalized_bboxes[..., 8:9]
        vy = normalized_bboxes[..., 9:10]
        return torch.cat((cx, cy, cz, w, l, h, rot, vx, vy), dim=-1)
    return torch.cat((cx, cy, cz, w, l, h, rot), dim=-1)


def topk_gather(feat: torch.Tensor, topk_indexes: torch.Tensor | None) -> torch.Tensor:
    """Gather along dim=1 using per-batch top-k indices."""
    return gather_dim1_topk(feat, topk_indexes)


def memory_refresh(memory: torch.Tensor, prev_exist: torch.Tensor) -> torch.Tensor:
    """Reset memory for new scenes with prev_exist mask in [B]."""

    memory_shape = memory.shape
    view_shape = [1 for _ in range(len(memory_shape))]
    prev_exist = prev_exist.view(-1, *view_shape[1:])
    return memory * prev_exist


_STREAMPETR_META_PROFILE = CameraMetaProfile(
    required_keys=("img_shape", "lidar2img"),
    matrix_exact_shapes={"lidar2img": (4, 4)},
)


def validate_streampetr_img_metas(
    img_metas: list[dict[str, Any]],
    *,
    batch_size: int | None = None,
    num_cams: int | None = None,
    require_scene_token: bool = False,
) -> None:
    """Validate the subset of `img_metas` contract used by StreamPETR forward."""
    profile = CameraMetaProfile(
        required_keys=_STREAMPETR_META_PROFILE.required_keys,
        matrix_exact_shapes=_STREAMPETR_META_PROFILE.matrix_exact_shapes,
        require_scene_token=require_scene_token,
    )
    validate_camera_img_metas(
        img_metas,
        profile=profile,
        batch_size=batch_size,
        num_cams=num_cams,
    )


def build_img2lidars(
    img_metas: list[dict[str, Any]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    num_cams: int,
) -> torch.Tensor:
    """Build [B, Ncam, 4, 4] inverse projection matrices from metadata."""
    validate_streampetr_img_metas(img_metas, batch_size=len(img_metas), num_cams=num_cams)
    return build_img2lidars_from_metas(
        img_metas,
        device=device,
        dtype=dtype,
        num_cams=num_cams,
    )



