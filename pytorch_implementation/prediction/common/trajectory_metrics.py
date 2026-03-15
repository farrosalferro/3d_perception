"""Shared trajectory metrics for prediction models."""

from __future__ import annotations

import torch


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

    if pred_xy.shape != target_xy.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred_xy.shape)} target={tuple(target_xy.shape)}")
    if pred_xy.dim() != 3 or pred_xy.shape[-1] != 2:
        raise ValueError("Expected [B, T, 2] trajectory tensors.")

    batch_size, horizon, _ = pred_xy.shape
    if valid_mask is None:
        mask = torch.ones((batch_size, horizon), device=pred_xy.device, dtype=pred_xy.dtype)
    else:
        if valid_mask.shape != (batch_size, horizon):
            raise ValueError(f"Expected valid_mask shape {(batch_size, horizon)}, got {tuple(valid_mask.shape)}")
        mask = valid_mask.to(device=pred_xy.device, dtype=pred_xy.dtype)

    l2 = torch.linalg.norm(pred_xy - target_xy, dim=-1)
    denom = mask.sum(dim=-1).clamp_min(1.0)
    ade = (l2 * mask).sum(dim=-1) / denom

    last_valid_idx = (mask.sum(dim=-1).long() - 1).clamp_min(0)
    fde = l2.gather(dim=1, index=last_valid_idx.unsqueeze(-1)).squeeze(-1)
    return ade, fde


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

    if pred_modes.dim() != 4 or pred_modes.shape[-1] != 2:
        raise ValueError("Expected pred_modes shape [B, M, T, 2].")
    batch_size, num_modes, horizon, _ = pred_modes.shape
    if target_xy.shape != (batch_size, horizon, 2):
        raise ValueError(
            f"Expected target shape {(batch_size, horizon, 2)}, got {tuple(target_xy.shape)}"
        )

    expanded_target = target_xy.unsqueeze(1).expand(-1, num_modes, -1, -1)
    distances = torch.linalg.norm(pred_modes - expanded_target, dim=-1)

    if valid_mask is None:
        mask = torch.ones((batch_size, horizon), device=pred_modes.device, dtype=pred_modes.dtype)
    else:
        if valid_mask.shape != (batch_size, horizon):
            raise ValueError(f"Expected valid_mask shape {(batch_size, horizon)}, got {tuple(valid_mask.shape)}")
        mask = valid_mask.to(device=pred_modes.device, dtype=pred_modes.dtype)

    mask = mask.unsqueeze(1).expand(-1, num_modes, -1)
    denom = mask.sum(dim=-1).clamp_min(1.0)
    ade_per_mode = (distances * mask).sum(dim=-1) / denom

    best_mode_idx = ade_per_mode.argmin(dim=-1)
    gather_index = best_mode_idx.view(batch_size, 1, 1, 1).expand(-1, 1, horizon, 2)
    best_traj = pred_modes.gather(dim=1, index=gather_index).squeeze(1)
    best_ade = ade_per_mode.gather(dim=1, index=best_mode_idx.unsqueeze(-1)).squeeze(-1)
    return best_mode_idx, best_traj, best_ade


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

