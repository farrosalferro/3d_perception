"""Configuration objects for a pure-PyTorch GameFormer-style planning path."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..common.contracts import PlanningE2EConfig, replace_config as _replace_contract


@dataclass(frozen=True)
class GameFormerPlanningConfig:
    """Forward-only config for interactive game-theoretic planning."""

    name: str = "planning/gameformer"
    e2e: PlanningE2EConfig = field(
        default_factory=lambda: PlanningE2EConfig(
            name="planning/gameformer_base",
            model_key="planning/gameformer",
            hidden_dim=128,
            num_candidates=6,
            future_steps=12,
            history_steps=8,
        )
    )
    game_levels: int = 3
    num_heads: int = 4
    decoder_layers: int = 2
    dropout: float = 0.1
    interactive_gain: float = 0.25


def base_forward_config() -> GameFormerPlanningConfig:
    """Return default GameFormer-style planning config."""

    return GameFormerPlanningConfig()


def debug_forward_config(**overrides: object) -> GameFormerPlanningConfig:
    """Return a compact config for quick tests/notebooks."""

    cfg = GameFormerPlanningConfig(
        name="planning/gameformer_debug",
        e2e=PlanningE2EConfig(
            name="planning/gameformer_debug",
            model_key="planning/gameformer",
            history_steps=4,
            future_steps=6,
            num_agents=8,
            map_polylines=12,
            points_per_polyline=6,
            hidden_dim=64,
            num_candidates=4,
            dt=0.5,
            max_speed=18.0,
            max_accel=5.5,
            max_curvature=0.35,
            safety_margin=1.8,
        ),
        game_levels=2,
        num_heads=4,
        decoder_layers=2,
        dropout=0.0,
        interactive_gain=0.2,
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
    return GameFormerPlanningConfig(**values)
