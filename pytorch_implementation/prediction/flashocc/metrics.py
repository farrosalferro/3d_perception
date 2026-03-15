"""Lightweight trajectory metrics for prediction smoke tests."""

from __future__ import annotations

import torch

from ..common.trajectory_metrics import (
    average_displacement_error as _shared_average_displacement_error,
    final_displacement_error as _shared_final_displacement_error,
    trajectory_smoothness_l2 as _shared_trajectory_smoothness_l2,
)


def average_displacement_error(
    pred_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ADE over horizon for tensors shaped [..., H, 2]."""
    return _shared_average_displacement_error(pred_traj, gt_traj, valid_mask)


def final_displacement_error(
    pred_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute FDE at the last horizon step for tensors shaped [..., H, 2]."""
    return _shared_final_displacement_error(pred_traj, gt_traj, valid_mask)


def trajectory_smoothness_l2(traj: torch.Tensor) -> torch.Tensor:
    """Second-order smoothness penalty over trajectory horizon."""
    return _shared_trajectory_smoothness_l2(traj)

