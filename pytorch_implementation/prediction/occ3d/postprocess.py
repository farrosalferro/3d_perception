"""Post-processing helpers for Occ3D predictions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Occ3DPostProcessorLite:
    """Decode logits and trajectories into task-level prediction contracts."""

    occupancy_threshold: float = 0.5

    def decode(self, preds_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        occupancy_logits = preds_dict["occupancy_logits"]
        occupancy_probs = occupancy_logits.sigmoid()
        occupancy_binary = occupancy_probs >= self.occupancy_threshold

        trajectory_deltas = preds_dict["trajectory_deltas"]
        trajectories = preds_dict["trajectories"]
        speeds = torch.linalg.norm(trajectory_deltas, dim=-1)

        return {
            "occupancy_probs": occupancy_probs,
            "occupancy_binary": occupancy_binary,
            "trajectory_deltas": trajectory_deltas,
            "trajectories": trajectories,
            "speeds": speeds,
            "time_indices": preds_dict["time_indices"],
        }
