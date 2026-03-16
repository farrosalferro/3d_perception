"""Shared end-to-end planning contracts and tensor-key conventions."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class PlanningE2EConfig:
    """Common debug/runtime contract for planning-model forward paths."""

    name: str
    model_key: str
    history_steps: int = 8
    future_steps: int = 12
    num_agents: int = 12
    state_dim: int = 6
    map_polylines: int = 24
    points_per_polyline: int = 8
    map_feat_dim: int = 2
    hidden_dim: int = 128
    num_candidates: int = 6
    dt: float = 0.5
    max_speed: float = 20.0
    max_accel: float = 6.0
    max_curvature: float = 0.3
    safety_margin: float = 2.0

    def __post_init__(self) -> None:
        if self.history_steps <= 0:
            raise ValueError("history_steps must be positive.")
        if self.future_steps <= 0:
            raise ValueError("future_steps must be positive.")
        if self.num_agents <= 0:
            raise ValueError("num_agents must be positive.")
        if self.state_dim < 4:
            raise ValueError("state_dim must be >= 4 (x, y, vx, vy).")
        if self.map_polylines <= 0:
            raise ValueError("map_polylines must be positive.")
        if self.points_per_polyline <= 1:
            raise ValueError("points_per_polyline must be > 1.")
        if self.map_feat_dim <= 0:
            raise ValueError("map_feat_dim must be positive.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.num_candidates <= 0:
            raise ValueError("num_candidates must be positive.")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.max_speed <= 0.0:
            raise ValueError("max_speed must be positive.")
        if self.max_accel <= 0.0:
            raise ValueError("max_accel must be positive.")
        if self.max_curvature <= 0.0:
            raise ValueError("max_curvature must be positive.")
        if self.safety_margin <= 0.0:
            raise ValueError("safety_margin must be positive.")


PLANNING_OUTPUT_KEYS: tuple[str, ...] = (
    "candidate_trajectories",
    "candidate_scores",
    "selected_index",
    "selected_trajectory",
    "velocity",
    "acceleration",
    "curvature",
    "feasible_mask",
    "collision_free_mask",
    "safety_margin_violation",
    "time_stamps",
)


def build_time_stamps(
    *,
    future_steps: int,
    dt: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build monotonically increasing planning horizon timestamps `[T]`."""

    return torch.arange(1, future_steps + 1, device=device, dtype=dtype) * float(dt)


def replace_config(cfg: PlanningE2EConfig, **kwargs: object) -> PlanningE2EConfig:
    """Return a copied config with selected fields replaced."""

    return replace(cfg, **kwargs)
