"""Shared debug-batch builders for planning models/tests/notebooks."""

from __future__ import annotations

import torch

from .contracts import PlanningE2EConfig
from .interface import PlanningBatch


def build_debug_batch(
    cfg: PlanningE2EConfig,
    *,
    batch_size: int = 2,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> PlanningBatch:
    """Build deterministic synthetic planning inputs."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if device is None:
        device = torch.device("cpu")

    torch.manual_seed(7)
    time = torch.arange(cfg.history_steps, device=device, dtype=dtype).view(1, cfg.history_steps, 1)
    base_pos = torch.randn(batch_size, 1, 2, device=device, dtype=dtype)
    base_vel = torch.randn(batch_size, 1, 2, device=device, dtype=dtype) * 0.25
    ego_xy = base_pos + time * base_vel
    ego_vel = base_vel.expand(-1, cfg.history_steps, -1)
    yaw = torch.zeros(batch_size, cfg.history_steps, 1, device=device, dtype=dtype)
    yaw_rate = torch.zeros(batch_size, cfg.history_steps, 1, device=device, dtype=dtype)
    ego_history = torch.cat((ego_xy, ego_vel, yaw, yaw_rate), dim=-1)

    agent_states = torch.randn(batch_size, cfg.num_agents, max(cfg.state_dim, 6), device=device, dtype=dtype)
    # Keep agent speed realistic in debug batches.
    agent_states[..., 2:4] = agent_states[..., 2:4] * 0.5

    map_polylines = torch.randn(
        batch_size,
        cfg.map_polylines,
        cfg.points_per_polyline,
        cfg.map_feat_dim,
        device=device,
        dtype=dtype,
    )
    route_features = torch.randn(batch_size, 16, 4, device=device, dtype=dtype)
    return PlanningBatch(
        ego_history=ego_history,
        agent_states=agent_states,
        map_polylines=map_polylines,
        route_features=route_features,
    )
