"""Shared prediction-task runtime helpers."""

from .time_contracts import (
    build_time_axis,
    ensure_strictly_increasing,
    resolve_time_indices,
    time_deltas_from_indices,
)
from .trajectory_metrics import (
    average_displacement_error,
    compute_ade_fde,
    final_displacement_error,
    select_best_mode_by_ade,
    trajectory_smoothness_l2,
)

__all__ = [
    "average_displacement_error",
    "build_time_axis",
    "compute_ade_fde",
    "ensure_strictly_increasing",
    "final_displacement_error",
    "resolve_time_indices",
    "select_best_mode_by_ade",
    "time_deltas_from_indices",
    "trajectory_smoothness_l2",
]

