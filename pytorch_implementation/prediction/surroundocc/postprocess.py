"""Postprocess utilities for prediction/surroundocc."""

from __future__ import annotations

import torch


def decode_predictions(
    outputs: dict[str, torch.Tensor],
    *,
    occupancy_threshold: float = 0.5,
) -> list[dict[str, torch.Tensor]]:
    """Decode raw logits into occupancy masks and trajectories.

    Args:
        outputs: Model outputs from ``SurroundOccPredictionLite.forward``.
        occupancy_threshold: Occupied threshold used on occupied probability.
    """

    occupancy_logits = outputs["occupancy_logits"]
    trajectory = outputs["trajectory"]
    velocity = outputs["velocity"]

    if occupancy_logits.dim() != 6:
        raise ValueError(
            "Expected occupancy_logits shape [B, H, Cocc, D, Hb, Wb], "
            f"got {tuple(occupancy_logits.shape)}."
        )
    if trajectory.dim() != 4:
        raise ValueError(
            "Expected trajectory shape [B, Nagents, H, 2], "
            f"got {tuple(trajectory.shape)}."
        )

    probs = torch.softmax(occupancy_logits, dim=2)
    occupied_probs = probs[:, :, 1] if probs.shape[2] > 1 else probs[:, :, 0]
    occupied_mask = occupied_probs > float(occupancy_threshold)

    batch_results: list[dict[str, torch.Tensor]] = []
    for batch_idx in range(occupancy_logits.shape[0]):
        batch_results.append(
            {
                "occupancy": occupied_mask[batch_idx],
                "occupancy_prob": occupied_probs[batch_idx],
                "trajectory": trajectory[batch_idx],
                "velocity": velocity[batch_idx],
            }
        )
    return batch_results


def trajectory_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """Compute a small ADE/FDE metric smoke for trajectory outputs."""

    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target must share shape, got {tuple(prediction.shape)} vs {tuple(target.shape)}."
        )
    if prediction.dim() != 4 or prediction.shape[-1] != 2:
        raise ValueError(
            "Expected shape [B, Nagents, H, 2] for trajectory metrics, "
            f"got {tuple(prediction.shape)}."
        )

    displacement = torch.norm(prediction - target, dim=-1)  # [B, Nagents, H]
    ade = float(displacement.mean().item())
    fde = float(displacement[..., -1].mean().item())
    return {"ade": ade, "fde": fde}


def trajectory_consistency_error(
    trajectory: torch.Tensor,
    velocity: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Return temporal consistency error between displacement and velocity.

    This checks that ``trajectory[..., t] - trajectory[..., t-1]`` matches
    ``velocity[..., t] * dt`` for forecast steps ``t >= 1``.
    """

    if trajectory.shape != velocity.shape:
        raise ValueError(
            f"Trajectory and velocity must share shape, got {tuple(trajectory.shape)} vs {tuple(velocity.shape)}."
        )
    if trajectory.shape[2] < 2:
        return torch.tensor(0.0, device=trajectory.device, dtype=trajectory.dtype)
    step_displacement = trajectory[:, :, 1:, :] - trajectory[:, :, :-1, :]
    expected_displacement = velocity[:, :, 1:, :] * float(dt)
    return (step_displacement - expected_displacement).abs().amax()


def occupancy_iou(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Compute occupancy IoU between boolean occupancy tensors."""

    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target occupancy must share shape, got {tuple(prediction.shape)} vs {tuple(target.shape)}."
        )

    pred_bool = prediction.bool()
    target_bool = target.bool()
    intersection = (pred_bool & target_bool).sum(dtype=torch.float32)
    union = (pred_bool | target_bool).sum(dtype=torch.float32)
    if union.item() == 0.0:
        return 1.0
    return float((intersection / union).item())
