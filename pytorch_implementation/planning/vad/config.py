"""Configuration objects for a pure-PyTorch VAD-style planning forward path."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..common.contracts import PlanningE2EConfig, replace_config as _replace_contract


@dataclass(frozen=True)
class VADPlanningConfig:
    """Forward-only config for a vectorized-autonomous-driving planning module."""

    name: str = "planning/vad"
    e2e: PlanningE2EConfig = field(
        default_factory=lambda: PlanningE2EConfig(
            name="planning/vad_base",
            model_key="planning/vad",
            hidden_dim=128,
            num_candidates=8,
            future_steps=12,
            history_steps=8,
        )
    )
    lane_token_dim: int = 4
    interaction_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    boundary_weight: float = 1.0
    collision_weight: float = 1.0
    lane_align_weight: float = 1.0


def base_forward_config() -> VADPlanningConfig:
    """Return default VAD-style planning config."""

    return VADPlanningConfig()


def debug_forward_config(**overrides: object) -> VADPlanningConfig:
    """Return a compact config for quick tests/notebooks."""

    cfg = VADPlanningConfig(
        name="planning/vad_debug",
        e2e=PlanningE2EConfig(
            name="planning/vad_debug",
            model_key="planning/vad",
            history_steps=4,
            future_steps=6,
            num_agents=8,
            map_polylines=14,
            points_per_polyline=6,
            hidden_dim=64,
            num_candidates=5,
            dt=0.5,
            max_speed=18.0,
            max_accel=5.5,
            max_curvature=0.35,
            safety_margin=1.8,
        ),
        lane_token_dim=4,
        interaction_layers=2,
        num_heads=4,
        dropout=0.0,
    )
    if not overrides:
        return cfg

    values = dict(cfg.__dict__)
    e2e_overrides = overrides.pop("e2e_overrides", None)
    if e2e_overrides is not None:
        if not isinstance(e2e_overrides, dict):
            raise TypeError("e2e_overrides must be a dict when provided.")
        values["e2e"] = _replace_contract(cfg.e2e, **e2e_overrides)
    values.update(overrides)
    return VADPlanningConfig(**values)
