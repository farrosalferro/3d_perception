"""Temporal modules for BEVerse-lite trajectory forecasting."""

from __future__ import annotations

import torch
from torch import nn

from .config import BEVerseForwardConfig


class TemporalPredictorLite(nn.Module):
    """Decode a fixed horizon of temporal tokens from BEV context."""

    def __init__(self, cfg: BEVerseForwardConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.init_proj = nn.Linear(cfg.embed_dims, cfg.embed_dims)
        self.time_embedding = nn.Embedding(cfg.pred_horizon, cfg.embed_dims)
        self.gru = nn.GRU(input_size=cfg.embed_dims, hidden_size=cfg.embed_dims, batch_first=True)

    def forward(self, bev_embed: torch.Tensor) -> torch.Tensor:
        # bev_embed: [B, C, Hbev, Wbev]
        batch_size = bev_embed.shape[0]
        pooled = bev_embed.mean(dim=(-2, -1))
        h0 = torch.tanh(self.init_proj(pooled)).unsqueeze(0)
        time_ids = torch.arange(self.cfg.pred_horizon, device=bev_embed.device).unsqueeze(0)
        queries = self.time_embedding(time_ids.expand(batch_size, -1))
        temporal_tokens, _ = self.gru(queries, h0)
        return temporal_tokens
