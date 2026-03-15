"""Shared time-axis contract helpers for prediction models."""

from __future__ import annotations

from typing import Sequence

import torch


def ensure_strictly_increasing(indices: torch.Tensor, *, name: str) -> None:
    """Validate strict monotonicity along the last dimension."""

    if indices.shape[-1] <= 1:
        return
    diffs = indices[..., 1:] - indices[..., :-1]
    if not bool(torch.all(diffs > 0)):
        raise ValueError(f"{name} must be strictly increasing along the time axis.")


def build_time_axis(
    *,
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    dt: float = 1.0,
) -> torch.Tensor:
    """Build `[num_steps]` time axis as `arange(1..num_steps) * dt`."""

    return torch.arange(1, num_steps + 1, device=device, dtype=dtype) * float(dt)


def resolve_time_indices(
    time_indices: torch.Tensor | Sequence[float] | None,
    *,
    expected_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
    default_dt: float = 1.0,
    batch_size: int | None = None,
    require_strictly_increasing: bool = True,
) -> torch.Tensor:
    """Resolve optional time indices into validated 1D or 2D tensors."""

    if time_indices is None:
        base = build_time_axis(
            num_steps=expected_steps,
            device=device,
            dtype=dtype,
            dt=default_dt,
        )
        if batch_size is None:
            return base
        return base.unsqueeze(0).expand(batch_size, -1)

    tensor = (
        time_indices.to(device=device, dtype=dtype)
        if torch.is_tensor(time_indices)
        else torch.as_tensor(time_indices, device=device, dtype=dtype)
    )
    if tensor.dim() not in (1, 2):
        raise ValueError(f"{name} must be 1D or 2D, got {tuple(tensor.shape)}.")
    if tensor.shape[-1] != expected_steps:
        raise ValueError(f"{name} must have length={expected_steps}, got {tuple(tensor.shape)}.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values.")

    if batch_size is not None:
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0).expand(batch_size, -1)
        elif tensor.shape[0] != batch_size:
            raise ValueError(f"{name} must have shape ({batch_size}, {expected_steps}), got {tuple(tensor.shape)}.")
    if require_strictly_increasing:
        ensure_strictly_increasing(tensor, name=name)
    return tensor


def time_deltas_from_indices(time_indices: torch.Tensor) -> torch.Tensor:
    """Convert monotonic absolute indices into first-step + forward deltas."""

    if time_indices.dim() == 1:
        time_indices = time_indices.unsqueeze(0)
        squeeze = True
    elif time_indices.dim() == 2:
        squeeze = False
    else:
        raise ValueError(
            f"time_indices must be 1D or 2D, got {tuple(time_indices.shape)}."
        )
    deltas = torch.cat((time_indices[..., :1], time_indices[..., 1:] - time_indices[..., :-1]), dim=-1)
    return deltas.squeeze(0) if squeeze else deltas

