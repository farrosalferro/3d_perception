"""Trajectory prediction head for BEVerse-lite."""

from __future__ import annotations

import torch
from torch import nn

from .config import BEVerseForwardConfig


class TrajectoryHeadLite(nn.Module):
    """Predict multimodal future trajectories from temporal tokens."""

    def __init__(self, cfg: BEVerseForwardConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.shared = nn.Sequential(
            nn.LayerNorm(cfg.embed_dims),
            nn.Linear(cfg.embed_dims, cfg.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.embed_dims, cfg.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.delta_head = nn.Linear(cfg.embed_dims, cfg.num_modes * 2)
        self.mode_head = nn.Linear(cfg.embed_dims, cfg.num_modes)

    def forward(self, temporal_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        # temporal_tokens: [B, T, C]
        batch_size, horizon, _ = temporal_tokens.shape
        shared_tokens = self.shared(temporal_tokens)
        raw_deltas = self.delta_head(shared_tokens)
        raw_deltas = raw_deltas.view(batch_size, horizon, self.cfg.num_modes, 2)
        trajectory_deltas = torch.tanh(raw_deltas).permute(0, 2, 1, 3) * self.cfg.max_delta
        trajectory_preds = trajectory_deltas.cumsum(dim=2)

        mode_logits = self.mode_head(shared_tokens[:, -1, :])
        mode_probs = torch.softmax(mode_logits, dim=-1)
        return {
            "trajectory_deltas": trajectory_deltas,
            "trajectory_preds": trajectory_preds,
            "mode_logits": mode_logits,
            "mode_probs": mode_probs,
        }
