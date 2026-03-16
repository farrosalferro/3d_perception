"""Configuration objects for a pure-PyTorch UniAD-style planning forward path."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..common.contracts import PlanningE2EConfig, replace_config as _replace_contract


@dataclass(frozen=True)
class UniADPlanningConfig:
    """Forward-only config for an end-to-end UniAD-style planning module."""

    name: str = "planning/uniad"
    e2e: PlanningE2EConfig = field(
        default_factory=lambda: PlanningE2EConfig(
            name="planning/uniad_base",
            model_key="planning/uniad",
            hidden_dim=128,
            num_candidates=6,
            future_steps=12,
            history_steps=8,
        )
    )
    num_query_tokens: int = 32
    decoder_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    route_dim: int = 4


def base_forward_config() -> UniADPlanningConfig:
    """Return default UniAD-style planning config."""

    return UniADPlanningConfig()


def debug_forward_config(**overrides: object) -> UniADPlanningConfig:
    """Return a compact config for quick tests/notebooks."""

    cfg = UniADPlanningConfig(
        name="planning/uniad_debug",
        e2e=_replace_contract(
            PlanningE2EConfig(
                name="planning/uniad_debug",
                model_key="planning/uniad",
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
            )
        ),
        num_query_tokens=16,
        decoder_layers=2,
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
    return UniADPlanningConfig(**values)
