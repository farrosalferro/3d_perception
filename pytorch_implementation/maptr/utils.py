"""Math and positional-encoding helpers for MapTR."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn


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


class SinePositionalEncoding2D(nn.Module):
    """2D sine positional encoding for camera and BEV feature maps."""

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
