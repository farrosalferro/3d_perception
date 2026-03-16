"""Shared kinematic checks for planning trajectories."""

from __future__ import annotations

import torch


def _step_deltas(trajectories: torch.Tensor) -> torch.Tensor:
    if trajectories.ndim != 4 or trajectories.shape[-1] != 2:
        raise ValueError("trajectories must have shape [B, K, T, 2].")
    if trajectories.shape[2] < 2:
        return torch.zeros_like(trajectories)
    deltas = trajectories[:, :, 1:, :] - trajectories[:, :, :-1, :]
    first = deltas[:, :, :1, :]
    return torch.cat((first, deltas), dim=2)


def velocity_from_trajectory(trajectories: torch.Tensor, *, dt: float) -> torch.Tensor:
    """Estimate per-step velocity from trajectories `[B, K, T, 2]`."""

    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    return _step_deltas(trajectories) / float(dt)


def acceleration_from_velocity(velocity: torch.Tensor, *, dt: float) -> torch.Tensor:
    """Estimate per-step acceleration from velocity `[B, K, T, 2]`."""

    if velocity.ndim != 4 or velocity.shape[-1] != 2:
        raise ValueError("velocity must have shape [B, K, T, 2].")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if velocity.shape[2] < 2:
        return torch.zeros_like(velocity)
    deltas = velocity[:, :, 1:, :] - velocity[:, :, :-1, :]
    first = deltas[:, :, :1, :]
    return torch.cat((first, deltas), dim=2) / float(dt)


def curvature_from_trajectory(trajectories: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Compute scalar curvature magnitude per step for `[B, K, T, 2]`."""

    if trajectories.ndim != 4 or trajectories.shape[-1] != 2:
        raise ValueError("trajectories must have shape [B, K, T, 2].")
    if trajectories.shape[2] < 3:
        return trajectories.new_zeros(trajectories.shape[:-1])

    x = trajectories[..., 0]
    y = trajectories[..., 1]

    dx = x[:, :, 1:] - x[:, :, :-1]
    dy = y[:, :, 1:] - y[:, :, :-1]
    ddx = dx[:, :, 1:] - dx[:, :, :-1]
    ddy = dy[:, :, 1:] - dy[:, :, :-1]

    numerator = (dx[:, :, :-1] * ddy - dy[:, :, :-1] * ddx).abs()
    denom = (dx[:, :, :-1].square() + dy[:, :, :-1].square()).clamp(min=eps).pow(1.5)
    curv_mid = numerator / denom

    # Pad to [B, K, T] to keep key contracts simple for tests/docs.
    head = curv_mid[:, :, :1]
    return torch.cat((head, curv_mid, curv_mid[:, :, -1:]), dim=2)


def kinematic_feasible_mask(
    *,
    velocity: torch.Tensor,
    acceleration: torch.Tensor,
    curvature: torch.Tensor,
    max_speed: float,
    max_accel: float,
    max_curvature: float,
) -> torch.Tensor:
    """Return `[B, K]` boolean feasibility mask from hard kinematic bounds."""

    speed = torch.linalg.norm(velocity, dim=-1)
    accel = torch.linalg.norm(acceleration, dim=-1)
    feasible = (speed <= float(max_speed)).all(dim=-1)
    feasible &= (accel <= float(max_accel)).all(dim=-1)
    feasible &= (curvature <= float(max_curvature)).all(dim=-1)
    return feasible
