"""Trajectory metrics used by BEVerse-lite tests and decoding."""

from __future__ import annotations

import torch

from ..common.trajectory_metrics import (
    compute_ade_fde as _shared_compute_ade_fde,
    select_best_mode_by_ade as _shared_select_best_mode_by_ade,
)


def compute_ade_fde(
    pred_xy: torch.Tensor,
    target_xy: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ADE/FDE for batched trajectories.

    Args:
        pred_xy: [B, T, 2]
        target_xy: [B, T, 2]
        valid_mask: optional [B, T] float/bool mask (1 = valid)
    """

    return _shared_compute_ade_fde(pred_xy, target_xy, valid_mask)


def select_best_mode_by_ade(
    pred_modes: torch.Tensor,
    target_xy: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select best trajectory mode according to ADE.

    Args:
        pred_modes: [B, M, T, 2]
        target_xy: [B, T, 2]
        valid_mask: optional [B, T]
    Returns:
        best_mode_idx: [B]
        best_traj: [B, T, 2]
        best_ade: [B]
    """

    return _shared_select_best_mode_by_ade(pred_modes, target_xy, valid_mask)
