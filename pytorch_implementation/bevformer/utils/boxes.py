"""Bounding box encode/decode helpers."""

from __future__ import annotations

import torch


def normalize_bbox(bboxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes to the normalized representation used by BEVFormer."""

    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()
    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        return torch.cat((cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1)
    return torch.cat((cx, cy, w, l, cz, h, rot.sin(), rot.cos()), dim=-1)


def denormalize_bbox(normalized_bboxes: torch.Tensor) -> torch.Tensor:
    """Invert the BEVFormer normalized box representation."""

    rot = torch.atan2(normalized_bboxes[..., 6:7], normalized_bboxes[..., 7:8])
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]
    w = normalized_bboxes[..., 2:3].exp()
    l = normalized_bboxes[..., 3:4].exp()
    h = normalized_bboxes[..., 5:6].exp()
    if normalized_bboxes.size(-1) > 8:
        vx = normalized_bboxes[..., 8:9]
        vy = normalized_bboxes[..., 9:10]
        return torch.cat((cx, cy, cz, w, l, h, rot, vx, vy), dim=-1)
    return torch.cat((cx, cy, cz, w, l, h, rot), dim=-1)
