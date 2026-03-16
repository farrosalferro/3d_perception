"""Shared planning input/output tensor interfaces."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .contracts import PlanningE2EConfig


@dataclass
class PlanningBatch:
    """Container for common end-to-end planning inputs."""

    ego_history: torch.Tensor
    agent_states: torch.Tensor
    map_polylines: torch.Tensor
    route_features: torch.Tensor | None = None


def validate_planning_batch(batch: PlanningBatch, cfg: PlanningE2EConfig) -> None:
    """Validate common planning input shapes against config."""

    if batch.ego_history.ndim != 3 or batch.ego_history.shape[-1] != cfg.state_dim:
        raise ValueError(
            f"ego_history must be [B, {cfg.history_steps}, {cfg.state_dim}], "
            f"got {tuple(batch.ego_history.shape)}."
        )
    if batch.ego_history.shape[1] != cfg.history_steps:
        raise ValueError(
            f"ego_history time axis must be {cfg.history_steps}, got {batch.ego_history.shape[1]}."
        )
    if batch.agent_states.ndim != 3 or batch.agent_states.shape[1] != cfg.num_agents:
        raise ValueError(
            f"agent_states must be [B, {cfg.num_agents}, D], got {tuple(batch.agent_states.shape)}."
        )
    if batch.agent_states.shape[-1] < 4:
        raise ValueError("agent_states must include at least (x, y, vx, vy).")
    if batch.map_polylines.ndim != 4:
        raise ValueError(
            f"map_polylines must be [B, {cfg.map_polylines}, {cfg.points_per_polyline}, {cfg.map_feat_dim}], "
            f"got {tuple(batch.map_polylines.shape)}."
        )
    if batch.map_polylines.shape[1] != cfg.map_polylines:
        raise ValueError(
            f"map_polylines count must be {cfg.map_polylines}, got {batch.map_polylines.shape[1]}."
        )
    if batch.map_polylines.shape[2] != cfg.points_per_polyline:
        raise ValueError(
            "map_polylines point-count mismatch: "
            f"expected {cfg.points_per_polyline}, got {batch.map_polylines.shape[2]}."
        )
    if batch.map_polylines.shape[3] != cfg.map_feat_dim:
        raise ValueError(
            f"map_polylines feature dim must be {cfg.map_feat_dim}, got {batch.map_polylines.shape[3]}."
        )
    if batch.route_features is not None:
        if batch.route_features.ndim != 3:
            raise ValueError("route_features must be [B, R, F] when provided.")
        if batch.route_features.shape[0] != batch.ego_history.shape[0]:
            raise ValueError("route_features batch size must match ego_history batch size.")
