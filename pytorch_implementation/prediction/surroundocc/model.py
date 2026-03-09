"""Pure-PyTorch prediction/surroundocc forward path."""

from __future__ import annotations

import torch
from torch import nn

from .config import SurroundOccPredictionConfig
from .postprocess import decode_predictions


def _conv_norm_act(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),
    )


class SpatialEncoder(nn.Module):
    """Encode BEV history into compact spatial memory."""

    def __init__(self, in_channels: int, embed_dims: int) -> None:
        super().__init__()
        mid_channels = max(embed_dims // 2, 16)
        self.stem = _conv_norm_act(in_channels, mid_channels, stride=1)
        self.block1 = _conv_norm_act(mid_channels, mid_channels, stride=2)
        self.block2 = _conv_norm_act(mid_channels, embed_dims, stride=2)
        self.out_proj = _conv_norm_act(embed_dims, embed_dims, stride=1)

    def forward(self, history_bev: torch.Tensor) -> torch.Tensor:
        if history_bev.dim() != 5:
            raise ValueError(
                "Expected history_bev shape [B, T, C, Hb, Wb], "
                f"got {tuple(history_bev.shape)}."
            )
        batch_size, history_steps, channels, height, width = history_bev.shape
        x = history_bev.reshape(batch_size * history_steps, channels, height, width)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.out_proj(x)
        return x.view(batch_size, history_steps, x.shape[1], x.shape[2], x.shape[3])


class TemporalContextEncoder(nn.Module):
    """Encode temporal history with a recurrent context."""

    def __init__(self, embed_dims: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=embed_dims,
            hidden_size=embed_dims,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.norm = nn.LayerNorm(embed_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(self, spatial_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        temporal_tokens = spatial_features.mean(dim=(-1, -2))  # [B, T, C]
        sequence, _ = self.gru(temporal_tokens)
        sequence = self.dropout(self.norm(sequence))
        context = sequence[:, -1]
        return sequence, context


class HorizonDecoder(nn.Module):
    """Project temporal context into horizon-specific BEV features."""

    def __init__(self, embed_dims: int, future_steps: int, dropout: float) -> None:
        super().__init__()
        self.future_steps = int(future_steps)
        self.horizon_embedding = nn.Embedding(self.future_steps, embed_dims)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dims, embed_dims),
        )

    def forward(
        self,
        context: torch.Tensor,
        last_spatial: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = context.shape[0]
        horizon_index = torch.arange(self.future_steps, device=context.device)
        horizon_base = self.horizon_embedding(horizon_index)  # [H, C]
        horizon_tokens = context[:, None, :] + horizon_base[None, :, :]
        horizon_tokens = self.mlp(horizon_tokens)

        horizon_features = last_spatial[:, None, ...] + horizon_tokens[:, :, :, None, None]
        return horizon_features, horizon_tokens


class OccupancyHead(nn.Module):
    """Predict occupancy logits for each horizon and depth bin."""

    def __init__(self, embed_dims: int, occupancy_classes: int, depth_bins: int) -> None:
        super().__init__()
        self.occupancy_classes = int(occupancy_classes)
        self.depth_bins = int(depth_bins)
        self.refine = _conv_norm_act(embed_dims, embed_dims, stride=1)
        self.classifier = nn.Conv2d(
            embed_dims,
            self.occupancy_classes * self.depth_bins,
            kernel_size=1,
        )

    def forward(self, horizon_features: torch.Tensor) -> torch.Tensor:
        batch_size, future_steps, channels, height, width = horizon_features.shape
        x = horizon_features.reshape(batch_size * future_steps, channels, height, width)
        x = self.refine(x)
        x = self.classifier(x)
        x = x.view(
            batch_size,
            future_steps,
            self.occupancy_classes,
            self.depth_bins,
            height,
            width,
        )
        return x


class TrajectoryHead(nn.Module):
    """Predict agent trajectories with integrated velocity consistency."""

    def __init__(self, embed_dims: int, dropout: float) -> None:
        super().__init__()
        self.agent_proj = nn.Linear(4, embed_dims)
        self.delta_mlp = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dims, 2),
        )

    def forward(
        self,
        agent_states: torch.Tensor,
        horizon_tokens: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if agent_states.dim() != 3 or agent_states.shape[-1] != 4:
            raise ValueError(
                "Expected agent_states shape [B, Nagents, 4], "
                f"got {tuple(agent_states.shape)}."
            )
        agent_embed = self.agent_proj(agent_states)  # [B, Nagents, C]
        fused = agent_embed[:, :, None, :] + horizon_tokens[:, None, :, :]
        delta_velocity = 0.1 * self.delta_mlp(fused)

        base_velocity = agent_states[:, :, 2:4].unsqueeze(2)
        velocity = base_velocity + torch.cumsum(delta_velocity, dim=2)
        start_xy = agent_states[:, :, :2].unsqueeze(2)
        trajectory = start_xy + torch.cumsum(velocity * float(dt), dim=2)
        return trajectory, velocity, delta_velocity


class SurroundOccPredictionLite(nn.Module):
    """Standalone SurroundOcc-style occupancy + trajectory predictor."""

    def __init__(self, cfg: SurroundOccPredictionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.spatial_encoder = SpatialEncoder(cfg.in_channels, cfg.embed_dims)
        self.temporal_encoder = TemporalContextEncoder(cfg.embed_dims, cfg.dropout)
        self.horizon_decoder = HorizonDecoder(cfg.embed_dims, cfg.future_steps, cfg.dropout)
        self.occupancy_head = OccupancyHead(
            cfg.embed_dims,
            cfg.occupancy_classes,
            cfg.depth_bins,
        )
        self.trajectory_head = TrajectoryHead(cfg.embed_dims, cfg.dropout)

    def forward(
        self,
        history_bev: torch.Tensor,
        agent_states: torch.Tensor,
        *,
        decode: bool = False,
        occupancy_threshold: float = 0.5,
    ) -> dict[str, torch.Tensor] | dict[str, object]:
        if history_bev.shape[1] != self.cfg.history_steps:
            raise ValueError(
                f"Expected history_steps={self.cfg.history_steps}, got {history_bev.shape[1]}."
            )
        if history_bev.shape[2] != self.cfg.in_channels:
            raise ValueError(
                f"Expected in_channels={self.cfg.in_channels}, got {history_bev.shape[2]}."
            )
        if agent_states.shape[1] != self.cfg.num_agents:
            raise ValueError(
                f"Expected num_agents={self.cfg.num_agents}, got {agent_states.shape[1]}."
            )

        spatial_features = self.spatial_encoder(history_bev)
        temporal_sequence, temporal_context = self.temporal_encoder(spatial_features)
        horizon_features, horizon_tokens = self.horizon_decoder(
            temporal_context,
            spatial_features[:, -1],
        )
        occupancy_logits = self.occupancy_head(horizon_features)
        trajectory, velocity, delta_velocity = self.trajectory_head(
            agent_states,
            horizon_tokens,
            dt=self.cfg.dt,
        )

        outputs: dict[str, torch.Tensor] = {
            "occupancy_logits": occupancy_logits,
            "trajectory": trajectory,
            "velocity": velocity,
            "delta_velocity": delta_velocity,
            "horizon_tokens": horizon_tokens,
            "temporal_sequence": temporal_sequence,
        }
        if decode:
            decoded = decode_predictions(outputs, occupancy_threshold=occupancy_threshold)
            return {"preds": outputs, "decoded": decoded}
        return outputs
