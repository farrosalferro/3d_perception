"""Shared safety checks for planning trajectories."""

from __future__ import annotations

import torch


def rollout_agents_linear(
    agent_states: torch.Tensor,
    *,
    future_steps: int,
    dt: float,
) -> torch.Tensor:
    """Roll out agent centers with constant velocity.

    Args:
        agent_states: `[B, A, D]` where first 4 dims are `(x, y, vx, vy)`.
    Returns:
        `[B, A, T, 2]` predicted agent centers.
    """

    if agent_states.ndim != 3 or agent_states.shape[-1] < 4:
        raise ValueError("agent_states must have shape [B, A, D>=4].")
    if future_steps <= 0:
        raise ValueError("future_steps must be positive.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    pos = agent_states[..., :2].unsqueeze(2)
    vel = agent_states[..., 2:4].unsqueeze(2)
    time = (
        torch.arange(1, future_steps + 1, device=agent_states.device, dtype=agent_states.dtype)
        .view(1, 1, future_steps, 1)
    )
    return pos + vel * time * float(dt)


def pairwise_min_distance(
    candidate_trajectories: torch.Tensor,
    rolled_agents: torch.Tensor,
) -> torch.Tensor:
    """Return minimum distance `[B, K]` between ego candidates and agents."""

    if candidate_trajectories.ndim != 4 or candidate_trajectories.shape[-1] != 2:
        raise ValueError("candidate_trajectories must be [B, K, T, 2].")
    if rolled_agents.ndim != 4 or rolled_agents.shape[-1] != 2:
        raise ValueError("rolled_agents must be [B, A, T, 2].")
    if candidate_trajectories.shape[0] != rolled_agents.shape[0]:
        raise ValueError("Batch size mismatch between candidates and rolled agents.")
    if candidate_trajectories.shape[2] != rolled_agents.shape[2]:
        raise ValueError("Time axis mismatch between candidates and rolled agents.")

    # [B, K, A, T, 2]
    diff = candidate_trajectories[:, :, None, :, :] - rolled_agents[:, None, :, :, :]
    distance = torch.linalg.norm(diff, dim=-1)
    return distance.amin(dim=(-1, -2))


def collision_free_mask(
    *,
    candidate_trajectories: torch.Tensor,
    agent_states: torch.Tensor,
    safety_margin: float,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute collision-free mask and min-distance summary.

    Returns:
        - collision_free: `[B, K]` bool
        - min_distance: `[B, K]` float
    """

    rolled_agents = rollout_agents_linear(
        agent_states,
        future_steps=candidate_trajectories.shape[2],
        dt=dt,
    )
    min_distance = pairwise_min_distance(candidate_trajectories, rolled_agents)
    collision_free = min_distance >= float(safety_margin)
    return collision_free, min_distance
