"""Prediction metrics used by the Occ3D test contracts."""

from __future__ import annotations

import torch


def trajectory_ade_fde(
    pred_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute ADE/FDE over [B, A, T, 2] trajectories."""

    if pred_traj.shape != gt_traj.shape:
        raise ValueError(
            f"pred_traj and gt_traj must share shape, got {tuple(pred_traj.shape)} and {tuple(gt_traj.shape)}"
        )
    if pred_traj.dim() != 4 or pred_traj.shape[-1] != 2:
        raise ValueError(f"Expected [B, A, T, 2], got {tuple(pred_traj.shape)}")

    errors = torch.linalg.norm(pred_traj - gt_traj, dim=-1)
    if valid_mask is None:
        valid_mask = torch.ones_like(errors, dtype=torch.bool)
    if valid_mask.shape != errors.shape:
        raise ValueError(f"valid_mask must match [B, A, T], got {tuple(valid_mask.shape)}")

    valid = valid_mask.to(dtype=errors.dtype)
    ade = (errors * valid).sum() / valid.sum().clamp_min(1.0)
    final_valid = valid[..., -1]
    fde = (errors[..., -1] * final_valid).sum() / final_valid.sum().clamp_min(1.0)
    return {"ade": ade, "fde": fde}
