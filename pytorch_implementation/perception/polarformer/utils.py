"""Math and positional-encoding helpers for PolarFormer-lite."""

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


def _shape_hw(
    shape: Any,
    *,
    field_name: str,
    batch_idx: int,
    cam_idx: int,
) -> tuple[int, int]:
    if not isinstance(shape, (list, tuple)) or len(shape) < 2:
        raise ValueError(
            f"img_metas[{batch_idx}]['{field_name}'][{cam_idx}] must be a sequence"
            f" with at least (H, W), got {shape!r}."
        )
    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(
            f"img_metas[{batch_idx}]['{field_name}'][{cam_idx}] has non-positive size {(h, w)}."
        )
    return h, w


def _validate_matrix_shape(
    matrix: Any,
    *,
    field_name: str,
    batch_idx: int,
    cam_idx: int,
    expected_shape: tuple[int, int],
) -> None:
    matrix_tensor = torch.as_tensor(matrix)
    if tuple(matrix_tensor.shape) != expected_shape:
        raise ValueError(
            f"img_metas[{batch_idx}]['{field_name}'][{cam_idx}] must be {expected_shape}, "
            f"got {tuple(matrix_tensor.shape)}."
        )


def validate_polarformer_img_metas(
    img_metas: list[dict[str, Any]],
    *,
    batch_size: int | None = None,
    num_cams: int | None = None,
    strict_img_meta: bool = True,
    require_geometry: bool = True,
) -> None:
    """Validate the subset of `img_metas` contract used by PolarFormer forward."""

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
        if strict_img_meta and "pad_shape" not in meta:
            raise KeyError(f"img_metas[{batch_idx}] is missing required key 'pad_shape'.")
        if require_geometry:
            for field_name in ("lidar2img", "cam_intrinsic", "cam2lidar"):
                if field_name not in meta:
                    raise KeyError(f"img_metas[{batch_idx}] is missing required key '{field_name}'.")

        img_shapes = meta["img_shape"]
        pad_shapes = meta.get("pad_shape", img_shapes)
        if not isinstance(img_shapes, (list, tuple)):
            raise TypeError(f"img_metas[{batch_idx}]['img_shape'] must be a sequence, got {type(img_shapes)}.")
        if not isinstance(pad_shapes, (list, tuple)):
            raise TypeError(f"img_metas[{batch_idx}]['pad_shape'] must be a sequence, got {type(pad_shapes)}.")

        expected_num_cams = int(num_cams) if num_cams is not None else len(img_shapes)
        if len(img_shapes) != expected_num_cams:
            raise ValueError(
                f"img_metas[{batch_idx}]['img_shape'] has {len(img_shapes)} cameras, expected {expected_num_cams}."
            )
        if len(pad_shapes) != expected_num_cams:
            raise ValueError(
                f"img_metas[{batch_idx}]['pad_shape'] has {len(pad_shapes)} cameras, expected {expected_num_cams}."
            )

        for cam_idx in range(expected_num_cams):
            img_h, img_w = _shape_hw(
                img_shapes[cam_idx], field_name="img_shape", batch_idx=batch_idx, cam_idx=cam_idx
            )
            pad_h, pad_w = _shape_hw(
                pad_shapes[cam_idx], field_name="pad_shape", batch_idx=batch_idx, cam_idx=cam_idx
            )
            if strict_img_meta and (pad_h < img_h or pad_w < img_w):
                raise ValueError(
                    f"img_metas[{batch_idx}] pad_shape must be >= img_shape for cam {cam_idx}, "
                    f"got pad={(pad_h, pad_w)} and img={(img_h, img_w)}."
                )

        if require_geometry:
            lidar2img = meta["lidar2img"]
            cam_intrinsic = meta["cam_intrinsic"]
            cam2lidar = meta["cam2lidar"]
            for field_name, values in (
                ("lidar2img", lidar2img),
                ("cam_intrinsic", cam_intrinsic),
                ("cam2lidar", cam2lidar),
            ):
                if not isinstance(values, (list, tuple)):
                    raise TypeError(
                        f"img_metas[{batch_idx}]['{field_name}'] must be a sequence, got {type(values)}."
                    )
                if len(values) != expected_num_cams:
                    raise ValueError(
                        f"img_metas[{batch_idx}]['{field_name}'] has {len(values)} entries, "
                        f"expected {expected_num_cams}."
                    )

            for cam_idx in range(expected_num_cams):
                _validate_matrix_shape(
                    lidar2img[cam_idx],
                    field_name="lidar2img",
                    batch_idx=batch_idx,
                    cam_idx=cam_idx,
                    expected_shape=(4, 4),
                )
                _validate_matrix_shape(
                    cam2lidar[cam_idx],
                    field_name="cam2lidar",
                    batch_idx=batch_idx,
                    cam_idx=cam_idx,
                    expected_shape=(4, 4),
                )
                intr = torch.as_tensor(cam_intrinsic[cam_idx])
                if intr.ndim != 2 or intr.shape[0] < 3 or intr.shape[1] < 3:
                    raise ValueError(
                        f"img_metas[{batch_idx}]['cam_intrinsic'][{cam_idx}] must be at least 3x3, "
                        f"got {tuple(intr.shape)}."
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

    stacked: list[torch.Tensor] = []
    for batch_idx, meta in enumerate(img_metas):
        if field_name not in meta:
            raise KeyError(f"img_metas[{batch_idx}] is missing required key '{field_name}'.")
        values = meta[field_name]
        if not isinstance(values, (list, tuple)) or len(values) != num_cams:
            raise ValueError(
                f"img_metas[{batch_idx}]['{field_name}'] must have {num_cams} entries, got {values!r}."
            )
        per_cam = []
        for cam_idx in range(num_cams):
            matrix = torch.as_tensor(values[cam_idx], device=device, dtype=dtype)
            if tuple(matrix.shape) != expected_shape:
                raise ValueError(
                    f"img_metas[{batch_idx}]['{field_name}'][{cam_idx}] must be {expected_shape}, "
                    f"got {tuple(matrix.shape)}."
                )
            per_cam.append(matrix)
        stacked.append(torch.stack(per_cam, dim=0))
    return torch.stack(stacked, dim=0)


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


class SinePositionalEncoding2D(nn.Module):
    """2D sine positional encoding from a boolean mask."""

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

