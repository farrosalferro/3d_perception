"""Math, memory, and positional-encoding helpers for StreamPETR."""

from __future__ import annotations

import math

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

