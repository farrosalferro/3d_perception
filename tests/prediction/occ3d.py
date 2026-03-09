"""Intermediate tensor and prediction-contract tests for Occ3D."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from pytorch_implementation.prediction.occ3d.config import debug_forward_config
from pytorch_implementation.prediction.occ3d.metrics import trajectory_ade_fde
from pytorch_implementation.prediction.occ3d.model import Occ3DLite


def _first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _iter_tensors(value: Any):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)


def _register_hook(module, name: str, capture: dict[str, Any], handles: list) -> None:
    def _hook(_module, _inputs, output):
        capture[name] = output

    handles.append(module.register_forward_hook(_hook))


@pytest.fixture()
def instrumented_debug_forward():
    torch.manual_seed(7)
    cfg = debug_forward_config()
    model = Occ3DLite(cfg).eval()

    batch_size = 2
    bev_history = torch.randn(
        batch_size,
        cfg.history_frames,
        cfg.input_channels,
        cfg.bev_h,
        cfg.bev_w,
    )

    capture: dict[str, Any] = {}
    handles = []

    _register_hook(model.backbone.stem, "backbone.stem", capture, handles)
    _register_hook(model.backbone.block, "backbone.block", capture, handles)
    _register_hook(model.temporal_encoder.input_proj, "temporal_encoder.input_proj", capture, handles)
    _register_hook(model.temporal_encoder.gru, "temporal_encoder.gru", capture, handles)
    _register_hook(model.future_decoder.time_embedding, "future_decoder.time_embedding", capture, handles)
    _register_hook(model.future_decoder.gru, "future_decoder.gru", capture, handles)
    _register_hook(model.occupancy_head.proj, "occupancy_head.proj", capture, handles)
    _register_hook(model.trajectory_head.delta_proj, "trajectory_head.delta_proj", capture, handles)

    with torch.no_grad():
        outputs = model(bev_history, decode=False)

    for handle in handles:
        handle.remove()

    return {
        "cfg": cfg,
        "batch_size": batch_size,
        "bev_history": bev_history,
        "capture": capture,
        "outputs": outputs,
        "model": model,
    }


def test_intermediate_hooks_cover_critical_boundaries(instrumented_debug_forward):
    capture = instrumented_debug_forward["capture"]
    expected = {
        "backbone.stem",
        "backbone.block",
        "temporal_encoder.input_proj",
        "temporal_encoder.gru",
        "future_decoder.time_embedding",
        "future_decoder.gru",
        "occupancy_head.proj",
        "trajectory_head.delta_proj",
    }
    missing = sorted(expected - set(capture.keys()))
    assert not missing, f"Missing intermediate captures: {missing}"


def test_intermediate_shapes_match_debug_config(instrumented_debug_forward):
    cfg = instrumented_debug_forward["cfg"]
    batch_size = instrumented_debug_forward["batch_size"]
    capture = instrumented_debug_forward["capture"]
    outputs = instrumented_debug_forward["outputs"]

    flattened_batch = batch_size * cfg.history_frames
    assert _first_tensor(capture["backbone.stem"]).shape == (
        flattened_batch,
        cfg.embed_dims,
        cfg.bev_h,
        cfg.bev_w,
    )
    assert _first_tensor(capture["backbone.block"]).shape == (
        flattened_batch,
        cfg.embed_dims,
        cfg.bev_h,
        cfg.bev_w,
    )
    assert _first_tensor(capture["temporal_encoder.input_proj"]).shape == (
        batch_size,
        cfg.history_frames,
        cfg.temporal_hidden_dims,
    )
    assert _first_tensor(capture["temporal_encoder.gru"]).shape == (
        batch_size,
        cfg.history_frames,
        cfg.temporal_hidden_dims,
    )
    assert _first_tensor(capture["future_decoder.time_embedding"]).shape == (
        cfg.future_horizon,
        cfg.temporal_hidden_dims,
    )
    assert _first_tensor(capture["future_decoder.gru"]).shape == (
        batch_size,
        cfg.future_horizon,
        cfg.temporal_hidden_dims,
    )
    assert _first_tensor(capture["occupancy_head.proj"]).shape == (
        batch_size,
        cfg.future_horizon,
        cfg.bev_z * cfg.bev_h * cfg.bev_w,
    )
    assert _first_tensor(capture["trajectory_head.delta_proj"]).shape == (
        batch_size,
        cfg.future_horizon,
        cfg.num_agents * 2,
    )

    assert outputs["spatial_features"].shape == (
        batch_size,
        cfg.history_frames,
        cfg.embed_dims,
        cfg.bev_h,
        cfg.bev_w,
    )
    assert outputs["temporal_tokens"].shape == (
        batch_size,
        cfg.history_frames,
        cfg.temporal_hidden_dims,
    )
    assert outputs["temporal_encoded"].shape == (
        batch_size,
        cfg.history_frames,
        cfg.temporal_hidden_dims,
    )
    assert outputs["context"].shape == (batch_size, cfg.temporal_hidden_dims)
    assert outputs["future_states"].shape == (
        batch_size,
        cfg.future_horizon,
        cfg.temporal_hidden_dims,
    )
    assert outputs["occupancy_logits"].shape == (
        batch_size,
        cfg.future_horizon,
        cfg.bev_z,
        cfg.bev_h,
        cfg.bev_w,
    )
    assert outputs["trajectory_deltas"].shape == (
        batch_size,
        cfg.num_agents,
        cfg.future_horizon,
        2,
    )
    assert outputs["trajectories"].shape == (
        batch_size,
        cfg.num_agents,
        cfg.future_horizon,
        2,
    )
    assert outputs["time_indices"].shape == (batch_size, cfg.future_horizon)


def test_intermediate_and_final_tensors_are_finite(instrumented_debug_forward):
    capture = instrumented_debug_forward["capture"]
    outputs = instrumented_debug_forward["outputs"]

    for name, value in capture.items():
        tensors = list(_iter_tensors(value))
        assert tensors, f"No tensor found in captured output for '{name}'."
        for tensor in tensors:
            assert torch.isfinite(tensor).all(), f"Non-finite values found in intermediate '{name}'."

    for name, value in outputs.items():
        if value is None:
            continue
        tensors = list(_iter_tensors(value))
        assert tensors, f"No tensor found in output '{name}'."
        for tensor in tensors:
            assert torch.isfinite(tensor).all(), f"Non-finite values found in final output '{name}'."


def test_prediction_horizon_and_trajectory_consistency(instrumented_debug_forward):
    cfg = instrumented_debug_forward["cfg"]
    model = instrumented_debug_forward["model"]
    bev_history = instrumented_debug_forward["bev_history"]
    outputs = instrumented_debug_forward["outputs"]

    # Horizon / time-axis integrity.
    expected_axis = torch.arange(cfg.future_horizon).unsqueeze(0).expand(outputs["time_indices"].shape[0], -1)
    assert torch.equal(outputs["time_indices"], expected_axis)
    assert torch.all(outputs["time_indices"][:, 1:] - outputs["time_indices"][:, :-1] == 1)

    # Trajectory consistency: integrated positions must match cumsum of deltas.
    reconstructed = outputs["trajectory_deltas"].cumsum(dim=2)
    assert torch.allclose(outputs["trajectories"], reconstructed, atol=1e-5, rtol=1e-5)

    # Decode contract for prediction outputs.
    with torch.no_grad():
        decoded_out = model(bev_history, decode=True)
    decoded = decoded_out["decoded"]
    assert decoded["occupancy_probs"].shape == outputs["occupancy_logits"].shape
    assert decoded["occupancy_binary"].shape == outputs["occupancy_logits"].shape
    assert decoded["occupancy_binary"].dtype == torch.bool
    assert decoded["trajectories"].shape == outputs["trajectories"].shape
    assert decoded["trajectory_deltas"].shape == outputs["trajectory_deltas"].shape
    assert decoded["speeds"].shape == outputs["trajectory_deltas"].shape[:-1]
    assert decoded["time_indices"].shape == outputs["time_indices"].shape


def test_prediction_metric_smoke(instrumented_debug_forward):
    outputs = instrumented_debug_forward["outputs"]
    pred = outputs["trajectories"]
    gt = pred + 0.05
    metrics = trajectory_ade_fde(pred, gt)

    assert set(metrics.keys()) == {"ade", "fde"}
    for key, value in metrics.items():
        assert torch.isfinite(value), f"{key} is non-finite"
        assert float(value) >= 0.0
