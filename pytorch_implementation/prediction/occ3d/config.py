"""Configuration for the pure-PyTorch Occ3D prediction model."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Occ3DForwardConfig:
    """Forward-only configuration used by the educational Occ3D variant."""

    name: str
    history_frames: int = 4
    future_horizon: int = 6
    num_agents: int = 8
    input_channels: int = 16
    embed_dims: int = 64
    temporal_hidden_dims: int = 64
    bev_h: int = 20
    bev_w: int = 20
    bev_z: int = 4
    occupancy_threshold: float = 0.5


def occ3d_base_forward_config() -> Occ3DForwardConfig:
    """Default compact configuration for Occ3D-style occupancy prediction."""

    return Occ3DForwardConfig(name="occ3d_base_forward")


def debug_forward_config(
    *,
    history_frames: int = 3,
    future_horizon: int = 5,
    num_agents: int = 6,
    input_channels: int = 12,
    embed_dims: int = 48,
    temporal_hidden_dims: int = 48,
    bev_h: int = 16,
    bev_w: int = 16,
    bev_z: int = 3,
) -> Occ3DForwardConfig:
    """Small debug config for fast local tests."""

    base = occ3d_base_forward_config()
    return replace(
        base,
        name=f"{base.name}_debug",
        history_frames=int(history_frames),
        future_horizon=int(future_horizon),
        num_agents=int(num_agents),
        input_channels=int(input_channels),
        embed_dims=int(embed_dims),
        temporal_hidden_dims=int(temporal_hidden_dims),
        bev_h=int(bev_h),
        bev_w=int(bev_w),
        bev_z=int(bev_z),
    )


def replace_config(cfg: Occ3DForwardConfig, **kwargs: object) -> Occ3DForwardConfig:
    """Return a copied config with selected values replaced."""

    return replace(cfg, **kwargs)
