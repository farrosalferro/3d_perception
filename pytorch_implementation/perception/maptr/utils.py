"""Math and positional-encoding helpers for MapTR."""

from __future__ import annotations

import math
from typing import Sequence

import torch

from ...common.utils.numerics import inverse_sigmoid as _shared_inverse_sigmoid
from ...common.utils.positional_encoding import SinePositionalEncoding2D


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Numerically stable inverse-sigmoid used by iterative reference updates."""
    return _shared_inverse_sigmoid(x, eps=eps, strict_clamp=True)


def points_to_boxes(pts: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert normalized polyline points [B, V, P, 2] into [B, V, 4] cxcywh boxes."""

    if pts.dim() != 4 or pts.shape[-1] != 2:
        raise ValueError(f"pts must have shape [B, V, P, 2], got {tuple(pts.shape)}")
    pts_x = pts[..., 0]
    pts_y = pts[..., 1]
    xmin = pts_x.min(dim=-1).values
    xmax = pts_x.max(dim=-1).values
    ymin = pts_y.min(dim=-1).values
    ymax = pts_y.max(dim=-1).values
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    w = (xmax - xmin).clamp(min=eps)
    h = (ymax - ymin).clamp(min=eps)
    return torch.stack((cx, cy, w, h), dim=-1)


def denormalize_points(pts: torch.Tensor, pc_range: Sequence[float]) -> torch.Tensor:
    """Map normalized [0, 1] points to metric XY range using pc_range."""

    if len(pc_range) != 6:
        raise ValueError(f"pc_range must have 6 values, got {len(pc_range)}")
    out = pts.clone()
    out[..., 0] = out[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    out[..., 1] = out[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    return out


def denormalize_boxes_cxcywh(boxes: torch.Tensor, pc_range: Sequence[float]) -> torch.Tensor:
    """Map normalized cxcywh boxes to metric xyxy boxes."""

    if len(pc_range) != 6:
        raise ValueError(f"pc_range must have 6 values, got {len(pc_range)}")
    cx = boxes[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    cy = boxes[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    w = boxes[..., 2] * (pc_range[3] - pc_range[0])
    h = boxes[..., 3] * (pc_range[4] - pc_range[1])
    half_w = 0.5 * w
    half_h = 0.5 * h
    x1 = cx - half_w
    y1 = cy - half_h
    x2 = cx + half_w
    y2 = cy + half_h
    return torch.stack((x1, y1, x2, y2), dim=-1)


