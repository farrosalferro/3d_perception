"""Standalone pure-PyTorch Occ3D-style prediction model."""

from __future__ import annotations

import torch
from torch import nn

from .config import Occ3DForwardConfig
from .postprocess import Occ3DPostProcessorLite


class SpatialBackbone(nn.Module):
    """Spatial encoder over BEV feature maps."""

    def __init__(self, cfg: Occ3DForwardConfig) -> None:
        super().__init__()
        self.stem = nn.Conv2d(cfg.input_channels, cfg.embed_dims, kernel_size=3, padding=1)
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.embed_dims, cfg.embed_dims, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.embed_dims, cfg.embed_dims, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.stem(x))


class TemporalEncoder(nn.Module):
    """Encodes history frames into a compact temporal context."""

    def __init__(self, cfg: Occ3DForwardConfig) -> None:
        super().__init__()
        self.input_proj = nn.Linear(cfg.embed_dims, cfg.temporal_hidden_dims)
        self.gru = nn.GRU(
            input_size=cfg.temporal_hidden_dims,
            hidden_size=cfg.temporal_hidden_dims,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, spatial_feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # spatial_feats: [B, Th, C, H, W]
        pooled = spatial_feats.mean(dim=(-2, -1))
        tokens = self.input_proj(pooled)
        encoded, _ = self.gru(tokens)
        context = encoded[:, -1]
        return tokens, encoded, context


class FutureDecoder(nn.Module):
    """Autoregressive-lite temporal decoder for future horizon states."""

    def __init__(self, cfg: Occ3DForwardConfig) -> None:
        super().__init__()
        self.time_embedding = nn.Embedding(cfg.future_horizon, cfg.temporal_hidden_dims)
        self.gru = nn.GRU(
            input_size=cfg.temporal_hidden_dims,
            hidden_size=cfg.temporal_hidden_dims,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, context: torch.Tensor, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = context.shape[0]
        if horizon > self.time_embedding.num_embeddings:
            raise ValueError(
                f"horizon={horizon} exceeds decoder support {self.time_embedding.num_embeddings}"
            )
        time_steps = torch.arange(horizon, device=context.device)
        queries = self.time_embedding(time_steps).unsqueeze(0).expand(batch_size, -1, -1)
        queries = queries + context.unsqueeze(1)
        decoded, _ = self.gru(queries, context.unsqueeze(0))
        return decoded, time_steps


class OccupancyHead(nn.Module):
    """Predicts voxel occupancy logits for each future step."""

    def __init__(self, cfg: Occ3DForwardConfig) -> None:
        super().__init__()
        self.bev_z = cfg.bev_z
        self.bev_h = cfg.bev_h
        self.bev_w = cfg.bev_w
        self.proj = nn.Linear(cfg.temporal_hidden_dims, cfg.bev_z * cfg.bev_h * cfg.bev_w)

    def forward(self, future_states: torch.Tensor) -> torch.Tensor:
        batch_size, horizon, _ = future_states.shape
        logits = self.proj(future_states)
        return logits.view(batch_size, horizon, self.bev_z, self.bev_h, self.bev_w)


class TrajectoryHead(nn.Module):
    """Predicts per-step displacements and integrated trajectories."""

    def __init__(self, cfg: Occ3DForwardConfig) -> None:
        super().__init__()
        self.num_agents = cfg.num_agents
        self.delta_proj = nn.Linear(cfg.temporal_hidden_dims, cfg.num_agents * 2)

    def forward(self, future_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, horizon, _ = future_states.shape
        deltas = self.delta_proj(future_states).view(batch_size, horizon, self.num_agents, 2)
        deltas = deltas.permute(0, 2, 1, 3).contiguous()
        trajectories = deltas.cumsum(dim=2)
        return deltas, trajectories


class Occ3DLite(nn.Module):
    """Educational Occ3D-style predictor with occupancy and trajectory outputs."""

    def __init__(self, cfg: Occ3DForwardConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = SpatialBackbone(cfg)
        self.temporal_encoder = TemporalEncoder(cfg)
        self.future_decoder = FutureDecoder(cfg)
        self.occupancy_head = OccupancyHead(cfg)
        self.trajectory_head = TrajectoryHead(cfg)
        self.postprocess = Occ3DPostProcessorLite(occupancy_threshold=cfg.occupancy_threshold)

    def forward(
        self,
        bev_history: torch.Tensor,
        *,
        decode: bool = False,
    ) -> dict[str, torch.Tensor] | dict[str, object]:
        # bev_history: [B, Th, Cin, H, W]
        if bev_history.dim() != 5:
            raise ValueError(f"Expected [B, Th, C, H, W], got {tuple(bev_history.shape)}")
        batch_size, history_frames, in_channels, height, width = bev_history.shape
        if history_frames != self.cfg.history_frames:
            raise ValueError(
                f"history_frames mismatch, expected {self.cfg.history_frames} got {history_frames}"
            )
        if in_channels != self.cfg.input_channels:
            raise ValueError(f"input_channels mismatch, expected {self.cfg.input_channels} got {in_channels}")
        if height != self.cfg.bev_h or width != self.cfg.bev_w:
            raise ValueError(
                f"BEV size mismatch, expected ({self.cfg.bev_h}, {self.cfg.bev_w}) got ({height}, {width})"
            )

        spatial_flat = self.backbone(bev_history.view(batch_size * history_frames, in_channels, height, width))
        spatial_feats = spatial_flat.view(batch_size, history_frames, self.cfg.embed_dims, height, width)

        temporal_tokens, temporal_encoded, context = self.temporal_encoder(spatial_feats)
        future_states, time_steps = self.future_decoder(context, self.cfg.future_horizon)
        occupancy_logits = self.occupancy_head(future_states)
        trajectory_deltas, trajectories = self.trajectory_head(future_states)

        time_indices = time_steps.unsqueeze(0).expand(batch_size, -1)
        outputs = {
            "spatial_features": spatial_feats,
            "temporal_tokens": temporal_tokens,
            "temporal_encoded": temporal_encoded,
            "context": context,
            "future_states": future_states,
            "occupancy_logits": occupancy_logits,
            "trajectory_deltas": trajectory_deltas,
            "trajectories": trajectories,
            "time_indices": time_indices,
        }
        if decode:
            return {"preds": outputs, "decoded": self.postprocess.decode(outputs)}
        return outputs
