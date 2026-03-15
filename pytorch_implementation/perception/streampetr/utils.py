"""Math, metadata, memory, and positional-encoding helpers for StreamPETR."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import nn


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Numerically stable inverse sigmoid."""

    x = x.clamp(min=0.0, max=1.0)
    x1 = x.clamp(min=eps)
    x2 = (1.0 - x).clamp(min=eps)
    return torch.log(x1 / x2)


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

    if topk_indexes is None:
        return feat
    feat_shape = feat.shape
    topk_shape = topk_indexes.shape
    view_shape = [1 for _ in range(len(feat_shape))]
    view_shape[:2] = topk_shape[:2]
    topk_indexes = topk_indexes.view(*view_shape)
    return torch.gather(feat, 1, topk_indexes.repeat(1, 1, *feat_shape[2:]))


def memory_refresh(memory: torch.Tensor, prev_exist: torch.Tensor) -> torch.Tensor:
    """Reset memory for new scenes with prev_exist mask in [B]."""

    memory_shape = memory.shape
    view_shape = [1 for _ in range(len(memory_shape))]
    prev_exist = prev_exist.view(-1, *view_shape[1:])
    return memory * prev_exist


def _shape_hw(
    shape: Any,
    *,
    field_name: str,
    batch_idx: int,
    cam_idx: int,
) -> tuple[int, int]:
    if not isinstance(shape, (list, tuple)) or len(shape) < 2:
        raise ValueError(
            f"img_metas[{batch_idx}]['{field_name}'][{cam_idx}] must be a sequence with at least (H, W), got {shape!r}."
        )
    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"img_metas[{batch_idx}]['{field_name}'][{cam_idx}] has non-positive size {(h, w)}.")
    return h, w


def validate_streampetr_img_metas(
    img_metas: list[dict[str, Any]],
    *,
    batch_size: int | None = None,
    num_cams: int | None = None,
    require_scene_token: bool = False,
) -> None:
    """Validate the subset of `img_metas` contract used by StreamPETR forward."""

    if not isinstance(img_metas, list):
        raise TypeError(f"img_metas must be a list[dict], got {type(img_metas)}.")
    if not img_metas:
        raise ValueError("img_metas must not be empty.")
    if batch_size is not None and len(img_metas) != batch_size:
        raise ValueError(f"img_metas length {len(img_metas)} does not match batch size {batch_size}.")

    for batch_idx, meta in enumerate(img_metas):
        if not isinstance(meta, dict):
            raise TypeError(f"img_metas[{batch_idx}] must be a dict, got {type(meta)}.")
        if "img_shape" not in meta:
            raise KeyError(f"img_metas[{batch_idx}] is missing required key 'img_shape'.")
        if "lidar2img" not in meta:
            raise KeyError(f"img_metas[{batch_idx}] is missing required key 'lidar2img'.")
        if require_scene_token and "scene_token" not in meta:
            raise KeyError(
                f"img_metas[{batch_idx}] must include 'scene_token' when prev_exists is not provided explicitly."
            )

        img_shapes = meta["img_shape"]
        pad_shapes = meta.get("pad_shape", img_shapes)
        lidar2img = meta["lidar2img"]
        if not isinstance(img_shapes, (list, tuple)):
            raise TypeError(f"img_metas[{batch_idx}]['img_shape'] must be a sequence, got {type(img_shapes)}.")
        if not isinstance(pad_shapes, (list, tuple)):
            raise TypeError(f"img_metas[{batch_idx}]['pad_shape'] must be a sequence, got {type(pad_shapes)}.")
        if not isinstance(lidar2img, (list, tuple)):
            raise TypeError(f"img_metas[{batch_idx}]['lidar2img'] must be a sequence, got {type(lidar2img)}.")

        expected_num_cams = int(num_cams) if num_cams is not None else len(img_shapes)
        if len(img_shapes) != expected_num_cams:
            raise ValueError(
                f"img_metas[{batch_idx}]['img_shape'] has {len(img_shapes)} cameras, expected {expected_num_cams}."
            )
        if len(pad_shapes) != expected_num_cams:
            raise ValueError(
                f"img_metas[{batch_idx}]['pad_shape'] has {len(pad_shapes)} cameras, expected {expected_num_cams}."
            )
        if len(lidar2img) != expected_num_cams:
            raise ValueError(
                f"img_metas[{batch_idx}]['lidar2img'] has {len(lidar2img)} matrices, expected {expected_num_cams}."
            )

        for cam_idx in range(expected_num_cams):
            _shape_hw(img_shapes[cam_idx], field_name="img_shape", batch_idx=batch_idx, cam_idx=cam_idx)
            _shape_hw(pad_shapes[cam_idx], field_name="pad_shape", batch_idx=batch_idx, cam_idx=cam_idx)
            cam_mat = torch.as_tensor(lidar2img[cam_idx])
            if cam_mat.shape != (4, 4):
                raise ValueError(
                    f"img_metas[{batch_idx}]['lidar2img'][{cam_idx}] must be 4x4, got {tuple(cam_mat.shape)}."
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
    img2lidars: list[torch.Tensor] = []
    for batch_idx, meta in enumerate(img_metas):
        matrices = []
        for cam_idx, lidar2img in enumerate(meta["lidar2img"]):
            cam_mat = torch.as_tensor(lidar2img, device=device, dtype=dtype)
            if cam_mat.shape != (4, 4):
                raise ValueError(
                    f"img_metas[{batch_idx}]['lidar2img'][{cam_idx}] must be 4x4, got {tuple(cam_mat.shape)}."
                )
            matrices.append(torch.linalg.inv(cam_mat))
        img2lidars.append(torch.stack(matrices, dim=0))
    return torch.stack(img2lidars, dim=0)


class SinePositionalEncoding2D(nn.Module):
    """2D sine positional encoding for camera feature maps."""

    def __init__(
        self,
        num_feats: int,
        *,
        temperature: int = 10000,
        normalize: bool = True,
        scale: float = 2.0 * math.pi,
    ) -> None:
        super().__init__()
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """Return [B, 2*num_feats, H, W] encoding from [B, H, W] mask."""

        if mask.dim() != 3:
            raise ValueError(f"mask must have shape [B, H, W], got {tuple(mask.shape)}")
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)

        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_feats)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        pos = torch.cat((pos_y, pos_x), dim=-1)
        return pos.permute(0, 3, 1, 2).contiguous()

