"""Configuration objects for a pure-PyTorch VIP3D-style forward path."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class VIP3DConfig:
    """Forward-only trajectory prediction configuration."""

    name: str = "vip3d_base_forward"
    agent_input_dim: int = 4
    map_input_dim: int = 2
    hidden_dim: int = 128
    decoder_hidden_dim: int = 256
    num_heads: int = 4
    encoder_layers: int = 2
    num_modes: int = 6
    history_steps: int = 8
    future_steps: int = 12
    map_tokens: int = 24
    map_points_per_token: int = 6
    dropout: float = 0.1
    memory_bank_len: int = 8
    memory_bank_score_thresh: float = 0.0
    memory_bank_save_period: int = 3
    interaction_hidden_dim: int = 256
    interaction_dropout: float = 0.0
    interaction_random_drop: float = 0.0
    interaction_fp_ratio: float = 0.0
    interaction_update_query_pos: bool = True
    metadata_time_gap_reset: float = 10.0
    trajectory_relative: bool = True
    enforce_strict_metadata: bool = True
    pc_range: tuple[float, float, float, float, float, float] = (
        -51.2,
        -51.2,
        -5.0,
        51.2,
        51.2,
        3.0,
    )
    post_center_range: tuple[float, float, float, float, float, float] = (
        -61.2,
        -61.2,
        -10.0,
        61.2,
        61.2,
        10.0,
    )
    decode_max_num: int = 100
    decode_score_threshold: float = 0.2


def base_forward_config(**overrides: object) -> VIP3DConfig:
    """Create a base config with optional field overrides."""

    return replace(VIP3DConfig(), **overrides)


def debug_forward_config(**overrides: object) -> VIP3DConfig:
    """Create a compact deterministic config for tests and notebooks."""

    cfg = VIP3DConfig(
        name="vip3d_debug_forward",
        hidden_dim=64,
        decoder_hidden_dim=128,
        num_heads=4,
        encoder_layers=1,
        num_modes=3,
        history_steps=4,
        future_steps=6,
        map_tokens=8,
        map_points_per_token=4,
        dropout=0.0,
        memory_bank_len=4,
        interaction_hidden_dim=128,
        interaction_dropout=0.0,
        interaction_random_drop=0.0,
    )
    return replace(cfg, **overrides)
