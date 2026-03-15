"""Shared camera metadata validation helpers with profile adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class CameraMetaProfile:
    """Adapter profile for model-specific metadata contracts."""

    required_keys: tuple[str, ...] = ("img_shape", "lidar2img")
    require_pad_shape: bool = False
    enforce_pad_greater_equal_img: bool = False
    require_scene_token: bool = False
    matrix_exact_shapes: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    matrix_min_shapes: Mapping[str, tuple[int, int]] = field(default_factory=dict)


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


def validate_camera_img_metas(
    img_metas: list[dict[str, Any]],
    *,
    profile: CameraMetaProfile,
    batch_size: int | None = None,
    num_cams: int | None = None,
) -> None:
    """Validate camera-centric metadata using a profile adapter."""

    if not isinstance(img_metas, list):
        raise TypeError(f"img_metas must be a list[dict], got {type(img_metas)}.")
    if not img_metas:
        raise ValueError("img_metas must not be empty.")
    if batch_size is not None and len(img_metas) != batch_size:
        raise ValueError(f"img_metas length {len(img_metas)} does not match batch size {batch_size}.")

    required_keys = set(profile.required_keys)
    if profile.require_pad_shape:
        required_keys.add("pad_shape")

    for batch_idx, meta in enumerate(img_metas):
        if not isinstance(meta, dict):
            raise TypeError(f"img_metas[{batch_idx}] must be a dict, got {type(meta)}.")
        if profile.require_scene_token and "scene_token" not in meta:
            raise KeyError(
                f"img_metas[{batch_idx}] must include 'scene_token' for this metadata profile."
            )
        for key in sorted(required_keys):
            if key not in meta:
                raise KeyError(f"img_metas[{batch_idx}] is missing required key '{key}'.")

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
            if profile.enforce_pad_greater_equal_img and (pad_h < img_h or pad_w < img_w):
                raise ValueError(
                    f"img_metas[{batch_idx}] pad_shape must be >= img_shape for cam {cam_idx}, "
                    f"got pad={(pad_h, pad_w)} and img={(img_h, img_w)}."
                )

        for field_name, expected_shape in profile.matrix_exact_shapes.items():
            if field_name not in meta:
                raise KeyError(f"img_metas[{batch_idx}] is missing required key '{field_name}'.")
            values = meta[field_name]
            if not isinstance(values, (list, tuple)):
                raise TypeError(
                    f"img_metas[{batch_idx}]['{field_name}'] must be a sequence, got {type(values)}."
                )
            if len(values) != expected_num_cams:
                raise ValueError(
                    f"img_metas[{batch_idx}]['{field_name}'] has {len(values)} entries, "
                    f"expected {expected_num_cams}."
                )
            for cam_idx, matrix in enumerate(values):
                matrix_tensor = torch.as_tensor(matrix)
                if tuple(matrix_tensor.shape) != expected_shape:
                    raise ValueError(
                        f"img_metas[{batch_idx}]['{field_name}'][{cam_idx}] must be {expected_shape}, "
                        f"got {tuple(matrix_tensor.shape)}."
                    )

        for field_name, min_shape in profile.matrix_min_shapes.items():
            if field_name not in meta:
                raise KeyError(f"img_metas[{batch_idx}] is missing required key '{field_name}'.")
            values = meta[field_name]
            if not isinstance(values, (list, tuple)):
                raise TypeError(
                    f"img_metas[{batch_idx}]['{field_name}'] must be a sequence, got {type(values)}."
                )
            if len(values) != expected_num_cams:
                raise ValueError(
                    f"img_metas[{batch_idx}]['{field_name}'] has {len(values)} entries, "
                    f"expected {expected_num_cams}."
                )
            for cam_idx, matrix in enumerate(values):
                matrix_tensor = torch.as_tensor(matrix)
                if matrix_tensor.ndim != 2 or matrix_tensor.shape[0] < min_shape[0] or matrix_tensor.shape[1] < min_shape[1]:
                    raise ValueError(
                        f"img_metas[{batch_idx}]['{field_name}'][{cam_idx}] must be at least {min_shape}, "
                        f"got {tuple(matrix_tensor.shape)}."
                    )


def build_img2lidars_from_metas(
    img_metas: list[dict[str, Any]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    num_cams: int,
    lidar2img_field: str = "lidar2img",
) -> torch.Tensor:
    """Build `[B, Ncam, 4, 4]` inverse camera projection matrices from metadata."""

    img2lidars: list[torch.Tensor] = []
    for batch_idx, meta in enumerate(img_metas):
        if lidar2img_field not in meta:
            raise KeyError(f"img_metas[{batch_idx}] is missing required key '{lidar2img_field}'.")
        values = meta[lidar2img_field]
        if not isinstance(values, (list, tuple)) or len(values) != num_cams:
            raise ValueError(
                f"img_metas[{batch_idx}]['{lidar2img_field}'] must have {num_cams} entries, got {values!r}."
            )
        per_cam = []
        for cam_idx, lidar2img in enumerate(values):
            cam_mat = torch.as_tensor(lidar2img, device=device, dtype=dtype)
            if cam_mat.shape != (4, 4):
                raise ValueError(
                    f"img_metas[{batch_idx}]['{lidar2img_field}'][{cam_idx}] must be 4x4, "
                    f"got {tuple(cam_mat.shape)}."
                )
            per_cam.append(torch.linalg.inv(cam_mat))
        img2lidars.append(torch.stack(per_cam, dim=0))
    return torch.stack(img2lidars, dim=0)


def stack_camera_matrices(
    img_metas: list[dict[str, Any]],
    *,
    field_name: str,
    num_cams: int,
    device: torch.device,
    dtype: torch.dtype,
    expected_shape: tuple[int, int] | None = None,
    min_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Build `[B, Ncam, H, W]` camera matrix tensor from metadata."""

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
        for cam_idx, value in enumerate(values):
            matrix = torch.as_tensor(value, device=device, dtype=dtype)
            if expected_shape is not None and tuple(matrix.shape) != expected_shape:
                raise ValueError(
                    f"img_metas[{batch_idx}]['{field_name}'][{cam_idx}] must be {expected_shape}, "
                    f"got {tuple(matrix.shape)}."
                )
            if min_shape is not None and (
                matrix.ndim != 2 or matrix.shape[0] < min_shape[0] or matrix.shape[1] < min_shape[1]
            ):
                raise ValueError(
                    f"img_metas[{batch_idx}]['{field_name}'][{cam_idx}] must be at least {min_shape}, "
                    f"got {tuple(matrix.shape)}."
                )
            per_cam.append(matrix)
        stacked.append(torch.stack(per_cam, dim=0))
    return torch.stack(stacked, dim=0)

