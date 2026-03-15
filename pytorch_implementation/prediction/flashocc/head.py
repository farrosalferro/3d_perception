"""Trajectory prediction head for FlashOcc-style model."""

from __future__ import annotations

import torch
from torch import nn

from ...common.postprocess.gather import gather_mode_trajectories
from ..common.time_contracts import build_time_axis
from .config import FlashOccConfig


class FlashOccOccupancyHead(nn.Module):
    """BEVOCCHead2D-style occupancy logits head in pure torch."""

    def __init__(self, cfg: FlashOccConfig) -> None:
        super().__init__()
        ocfg = cfg.occupancy_head
        in_dim = cfg.backbone.embed_dims
        self.dz = int(ocfg.dz)
        self.num_classes = int(ocfg.num_classes)
        self.use_predicter = bool(ocfg.use_predicter)
        self.out_dim = int(ocfg.out_dim)

        out_channels = self.out_dim if self.use_predicter else self.num_classes * self.dz
        self.final_conv = nn.Conv2d(in_dim, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        if self.use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim * 2),
                nn.Softplus(),
                nn.Linear(self.out_dim * 2, self.num_classes * self.dz),
            )

    def forward(self, bev_feat: torch.Tensor) -> torch.Tensor:
        if bev_feat.dim() != 4:
            raise ValueError(f"Expected BEV features [B, C, Dy, Dx], got {tuple(bev_feat.shape)}")

        # Match upstream ordering: [B, C, Dy, Dx] -> [B, Dx, Dy, C].
        occ_pred = self.final_conv(bev_feat).permute(0, 3, 2, 1)
        batch_size, dx, dy, _ = occ_pred.shape
        if self.use_predicter:
            occ_pred = self.predicter(occ_pred)
        occ_pred = occ_pred.view(batch_size, dx, dy, self.dz, self.num_classes)
        return occ_pred


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
        self.occupancy_head = FlashOccOccupancyHead(cfg)

    def _build_time_axis(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return build_time_axis(
            num_steps=self.cfg.pred_horizon,
            device=device,
            dtype=dtype,
            dt=self.cfg.dt,
        )

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
        occupancy_logits = self.occupancy_head(bev_fused)
        occupancy_probs = occupancy_logits.softmax(dim=-1)
        best_mode_scores, best_mode_indices = mode_logits.softmax(dim=-1).max(dim=-1)
        best_trajectory = gather_mode_trajectories(
            traj_positions,
            best_mode_indices,
            mode_dim=2,
        ).squeeze(2)
        return {
            "query_tokens": query_tokens,
            "anchor_xy": anchor_xy,
            "traj_deltas": traj_deltas,
            "traj_positions": traj_positions,
            "traj_velocity": traj_velocity,
            "mode_logits": mode_logits,
            "best_mode_scores": best_mode_scores,
            "best_mode_indices": best_mode_indices,
            "best_trajectory": best_trajectory,
            "occupancy_logits": occupancy_logits,
            "occupancy_probs": occupancy_probs,
            "time_stamps": time_stamps,
        }

