"""Lightweight trajectory metrics for prediction smoke tests."""

from __future__ import annotations

import torch


def average_displacement_error(
    pred_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ADE over horizon for tensors shaped [..., H, 2]."""

    error = torch.linalg.norm(pred_traj - gt_traj, dim=-1)
    if valid_mask is not None:
        mask = valid_mask.to(dtype=error.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (error * mask).sum() / denom
    return error.mean()


def final_displacement_error(
    pred_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute FDE at the last horizon step for tensors shaped [..., H, 2]."""

    final_error = torch.linalg.norm(pred_traj[..., -1, :] - gt_traj[..., -1, :], dim=-1)
    if valid_mask is not None:
        mask = valid_mask[..., -1].to(dtype=final_error.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (final_error * mask).sum() / denom
    return final_error.mean()


def trajectory_smoothness_l2(traj: torch.Tensor) -> torch.Tensor:
    """Second-order smoothness penalty over trajectory horizon."""

    if traj.shape[-2] < 3:
        return torch.zeros((), dtype=traj.dtype, device=traj.device)
    accel = traj[..., 2:, :] - 2.0 * traj[..., 1:-1, :] + traj[..., :-2, :]
    return torch.linalg.norm(accel, dim=-1).mean()

