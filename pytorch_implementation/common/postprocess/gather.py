"""Shared gather helpers for decode and temporal-memory flows."""

from __future__ import annotations

import torch


def gather_dim1_topk(feat: torch.Tensor, topk_indexes: torch.Tensor | None) -> torch.Tensor:
    """Gather along dim=1 using per-batch top-k indices."""

    if topk_indexes is None:
        return feat
    feat_shape = feat.shape
    topk_shape = topk_indexes.shape
    view_shape = [1 for _ in range(len(feat_shape))]
    view_shape[:2] = topk_shape[:2]
    topk_indexes = topk_indexes.view(*view_shape)
    return torch.gather(feat, 1, topk_indexes.repeat(1, 1, *feat_shape[2:]))


def gather_mode_trajectories(
    trajectories: torch.Tensor,
    mode_indices: torch.Tensor,
    *,
    mode_dim: int = 1,
) -> torch.Tensor:
    """Gather trajectories along mode dimension.

    Supports:
      - trajectories [B, M, T, C] with mode_indices [B, K] at mode_dim=1
      - trajectories [B, Q, M, T, C] with mode_indices [B, Q] at mode_dim=2
    """

    if mode_dim not in (1, 2):
        raise ValueError(f"mode_dim must be 1 or 2, got {mode_dim}")

    if mode_dim == 1:
        if trajectories.ndim != 4:
            raise ValueError("mode_dim=1 expects trajectories shape [B, M, T, C].")
        batch_size, _, horizon, coord_dim = trajectories.shape
        if mode_indices.ndim == 1:
            mode_indices = mode_indices.unsqueeze(1)
        if mode_indices.ndim != 2 or mode_indices.shape[0] != batch_size:
            raise ValueError("mode_indices must have shape [B] or [B, K] for mode_dim=1.")
        gather_index = mode_indices.view(batch_size, -1, 1, 1).expand(-1, -1, horizon, coord_dim)
        return trajectories.gather(dim=1, index=gather_index)

    if trajectories.ndim != 5:
        raise ValueError("mode_dim=2 expects trajectories shape [B, Q, M, T, C].")
    batch_size, num_queries, _, horizon, coord_dim = trajectories.shape
    if mode_indices.ndim != 2 or mode_indices.shape != (batch_size, num_queries):
        raise ValueError("mode_indices must have shape [B, Q] for mode_dim=2.")
    gather_index = mode_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
        -1, -1, 1, horizon, coord_dim
    )
    return trajectories.gather(dim=2, index=gather_index)

