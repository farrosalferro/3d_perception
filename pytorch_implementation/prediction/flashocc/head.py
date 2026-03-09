"""Trajectory prediction head for FlashOcc-style model."""

from __future__ import annotations

import torch
from torch import nn

from .config import FlashOccConfig


class FlashOccPredictionHead(nn.Module):
    """Predict multi-modal trajectories from fused BEV features."""

    def __init__(self, cfg: FlashOccConfig) -> None:
        super().__init__()
        self.cfg = cfg
        embed_dims = cfg.backbone.embed_dims
        self.query_embed = nn.Parameter(torch.randn(cfg.num_queries, embed_dims) * 0.02)
        self.query_proj = nn.Linear(embed_dims, embed_dims)
        self.cross_attn = nn.MultiheadAttention(embed_dims, num_heads=cfg.num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.GELU(),
            nn.Linear(embed_dims, embed_dims),
        )
        self.norm = nn.LayerNorm(embed_dims)
        self.anchor_head = nn.Linear(embed_dims, 2)
        self.traj_head = nn.Linear(embed_dims, cfg.num_modes * cfg.pred_horizon * 2)
        self.mode_head = nn.Linear(embed_dims, cfg.num_modes)

    def _build_time_axis(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.arange(1, self.cfg.pred_horizon + 1, device=device, dtype=dtype) * self.cfg.dt

    def forward(self, bev_fused: torch.Tensor) -> dict[str, torch.Tensor]:
        if bev_fused.dim() != 4:
            raise ValueError(f"Expected fused BEV [B, C, H, W], got {tuple(bev_fused.shape)}")
        batch_size, embed_dims, _, _ = bev_fused.shape
        memory = bev_fused.flatten(2).transpose(1, 2)
        pooled = bev_fused.mean(dim=(-1, -2))
        query_tokens = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        query_tokens = query_tokens + self.query_proj(pooled).unsqueeze(1)
        attn_out, _ = self.cross_attn(query_tokens, memory, memory)
        query_tokens = self.norm(query_tokens + attn_out)
        query_tokens = self.norm(query_tokens + self.ffn(query_tokens))

        anchor_xy = self.anchor_head(query_tokens)
        traj_flat = self.traj_head(query_tokens)
        traj_deltas = traj_flat.view(
            batch_size,
            self.cfg.num_queries,
            self.cfg.num_modes,
            self.cfg.pred_horizon,
            2,
        )
        mode_logits = self.mode_head(query_tokens)
        traj_positions = anchor_xy.unsqueeze(2).unsqueeze(3) + torch.cumsum(traj_deltas, dim=3)
        traj_velocity = traj_deltas / self.cfg.dt
        time_stamps = self._build_time_axis(device=bev_fused.device, dtype=bev_fused.dtype)
        return {
            "query_tokens": query_tokens,
            "anchor_xy": anchor_xy,
            "traj_deltas": traj_deltas,
            "traj_positions": traj_positions,
            "traj_velocity": traj_velocity,
            "mode_logits": mode_logits,
            "time_stamps": time_stamps,
        }

