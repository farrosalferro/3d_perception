"""Postprocess utilities for prediction/surroundocc."""

from __future__ import annotations

import torch

from ..common.time_contracts import resolve_time_indices


def _resolve_future_time_indices(
    future_time_indices: torch.Tensor | None,
    *,
    batch_size: int,
    future_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if future_time_indices is not None and not torch.is_tensor(future_time_indices):
        raise TypeError("future_time_indices in outputs must be a tensor when provided.")
    return resolve_time_indices(
        future_time_indices,
        expected_steps=future_steps,
        device=device,
        dtype=dtype,
        name="future_time_indices",
        batch_size=batch_size,
        require_strictly_increasing=True,
    )


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
    future_time_indices = outputs.get("future_time_indices")

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
    if velocity.shape != trajectory.shape:
        raise ValueError(
            "Expected velocity shape to match trajectory shape "
            f"{tuple(trajectory.shape)}, got {tuple(velocity.shape)}."
        )
    if occupancy_logits.shape[1] != trajectory.shape[2]:
        raise ValueError(
            "Occupancy and trajectory future steps must match. "
            f"Got occupancy={occupancy_logits.shape[1]} trajectory={trajectory.shape[2]}."
        )
    if not torch.isfinite(occupancy_logits).all():
        raise ValueError("occupancy_logits must be finite before decode.")
    if not torch.isfinite(trajectory).all() or not torch.isfinite(velocity).all():
        raise ValueError("trajectory and velocity must be finite before decode.")

    batch_size, future_steps, num_classes, _, _, _ = occupancy_logits.shape
    resolved_future_times = _resolve_future_time_indices(
        future_time_indices,
        batch_size=batch_size,
        future_steps=future_steps,
        device=occupancy_logits.device,
        dtype=occupancy_logits.dtype,
    )

    if num_classes > 1:
        probs = torch.softmax(occupancy_logits, dim=2)
        occupied_probs = 1.0 - probs[:, :, 0]
        occupancy_semantic = probs.argmax(dim=2)
    else:
        probs = torch.sigmoid(occupancy_logits)
        occupied_probs = probs[:, :, 0]
        occupancy_semantic = None
    occupied_mask = occupied_probs > float(occupancy_threshold)

    batch_results: list[dict[str, torch.Tensor]] = []
    for batch_idx in range(batch_size):
        sample: dict[str, torch.Tensor] = {
            "occupancy": occupied_mask[batch_idx],
            "occupancy_prob": occupied_probs[batch_idx],
            "trajectory": trajectory[batch_idx],
            "velocity": velocity[batch_idx],
            "future_time_indices": resolved_future_times[batch_idx],
        }
        if occupancy_semantic is not None:
            sample["occupancy_semantic"] = occupancy_semantic[batch_idx]
        batch_results.append(sample)
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
    dt: float | torch.Tensor,
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
    if torch.is_tensor(dt):
        dt_tensor = dt.to(device=trajectory.device, dtype=trajectory.dtype)
        if dt_tensor.dim() == 1:
            if dt_tensor.shape[0] != trajectory.shape[2]:
                raise ValueError(
                    f"dt tensor must have length={trajectory.shape[2]}, got {tuple(dt_tensor.shape)}."
                )
            step_dt = dt_tensor[1:].view(1, 1, -1, 1)
        elif dt_tensor.dim() == 2:
            if dt_tensor.shape != (trajectory.shape[0], trajectory.shape[2]):
                raise ValueError(
                    "2D dt tensor must have shape "
                    f"{(trajectory.shape[0], trajectory.shape[2])}, got {tuple(dt_tensor.shape)}."
                )
            step_dt = dt_tensor[:, 1:].unsqueeze(1).unsqueeze(-1)
        else:
            raise ValueError(f"dt tensor must be 1D or 2D, got {tuple(dt_tensor.shape)}.")
    else:
        step_dt = trajectory.new_full((1, 1, trajectory.shape[2] - 1, 1), float(dt))
    step_displacement = trajectory[:, :, 1:, :] - trajectory[:, :, :-1, :]
    expected_displacement = velocity[:, :, 1:, :] * step_dt
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
