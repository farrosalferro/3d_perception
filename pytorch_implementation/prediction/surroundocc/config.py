"""Configuration objects for prediction/surroundocc."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple


@dataclass(frozen=True)
class SurroundOccPredictionConfig:
    """Forward configuration for a SurroundOcc-style predictor."""

    name: str = "prediction/surroundocc"
    history_steps: int = 4
    future_steps: int = 6
    num_agents: int = 8
    num_cams: int = 6
    in_channels: int = 32
    embed_dims: int = 128
    bev_hw: Tuple[int, int] = (32, 32)
    depth_bins: int = 6
    occupancy_classes: int = 2
    pc_range: Tuple[float, float, float, float, float, float] = (
        -50.0,
        -50.0,
        -5.0,
        50.0,
        50.0,
        3.0,
    )
    dt: float = 0.5
    dropout: float = 0.1


def surroundocc_prediction_config() -> SurroundOccPredictionConfig:
    """Return the default training-like forward config."""

    return SurroundOccPredictionConfig()


def debug_forward_config(
    *,
    history_steps: int = 4,
    future_steps: int = 5,
    num_agents: int = 6,
    in_channels: int = 16,
    embed_dims: int = 64,
    bev_hw: Tuple[int, int] = (24, 24),
    depth_bins: int = 4,
) -> SurroundOccPredictionConfig:
    """Return a compact config for quick local tests."""

    base = surroundocc_prediction_config()
    return replace(
        base,
        name=f"{base.name}_debug",
        history_steps=int(history_steps),
        future_steps=int(future_steps),
        num_agents=int(num_agents),
        in_channels=int(in_channels),
        embed_dims=int(embed_dims),
        bev_hw=(int(bev_hw[0]), int(bev_hw[1])),
        depth_bins=int(depth_bins),
    )
