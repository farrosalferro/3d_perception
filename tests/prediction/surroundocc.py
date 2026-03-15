"""Intermediate tensor validation tests for prediction/surroundocc."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from pytorch_implementation.prediction.surroundocc.config import debug_forward_config
from pytorch_implementation.prediction.surroundocc.model import SurroundOccPredictionLite
from pytorch_implementation.prediction.surroundocc.postprocess import (
    occupancy_iou,
    trajectory_consistency_error,
    trajectory_metrics,
)
from tests._shared.hook_helpers import register_hook_append
from tests._shared.tensor_helpers import conv2d_out, first_tensor, iter_tensors


def _first_tensor(value: Any) -> torch.Tensor | None:
    return first_tensor(value)


def _iter_tensors(value: Any):
    yield from iter_tensors(value)


def _register_hook(module, name: str, capture: dict[str, list[Any]], handles: list) -> None:
    register_hook_append(module, name, capture, handles)


def _conv2d_out(size: int, kernel: int, stride: int, padding: int) -> int:
    return conv2d_out(size, kernel, stride, padding)


@pytest.fixture()
def instrumented_debug_forward():
    cfg = debug_forward_config(
        history_steps=4,
        future_steps=5,
        num_agents=6,
        in_channels=16,
        embed_dims=64,
        bev_hw=(24, 24),
        depth_bins=4,
    )
    model = SurroundOccPredictionLite(cfg).eval()

    batch_size = 2
    history_bev = torch.randn(
        batch_size,
        cfg.history_steps,
        cfg.in_channels,
        cfg.bev_hw[0],
        cfg.bev_hw[1],
    )
    agent_states = torch.randn(batch_size, cfg.num_agents, 4)

    capture: dict[str, list[Any]] = {}
    handles = []
    _register_hook(model.spatial_encoder.stem, "spatial.stem", capture, handles)
    _register_hook(model.spatial_encoder.block1, "spatial.block1", capture, handles)
    _register_hook(model.spatial_encoder.block2, "spatial.block2", capture, handles)
    _register_hook(model.spatial_encoder.out_proj, "spatial.out_proj", capture, handles)
    _register_hook(model.temporal_encoder.gru, "temporal.gru", capture, handles)
    _register_hook(model.horizon_decoder.mlp, "horizon.mlp", capture, handles)
    _register_hook(model.occupancy_head.refine, "occupancy.refine", capture, handles)
    _register_hook(model.occupancy_head.classifier, "occupancy.classifier", capture, handles)
    _register_hook(model.trajectory_head.agent_proj, "trajectory.agent_proj", capture, handles)
    _register_hook(model.trajectory_head.delta_mlp, "trajectory.delta_mlp", capture, handles)

    with torch.no_grad():
        outputs = model(history_bev, agent_states, decode=False)
        decoded_outputs = model(history_bev, agent_states, decode=True)

    for handle in handles:
        handle.remove()

    decoded = decoded_outputs["decoded"]
    pred_traj = outputs["trajectory"]
    gt_traj = pred_traj + 0.05 * torch.randn_like(pred_traj)
    metric_smoke = trajectory_metrics(pred_traj, gt_traj)
    pred_occ = torch.stack([sample["occupancy"] for sample in decoded], dim=0)
    gt_occ = pred_occ.clone()
    gt_occ[..., 0, 0, 0] = True
    occ_iou = occupancy_iou(pred_occ, gt_occ)

    return {
        "cfg": cfg,
        "capture": capture,
        "outputs": outputs,
        "decoded": decoded,
        "model": model,
        "history_bev": history_bev,
        "agent_states": agent_states,
        "metric_smoke": metric_smoke,
        "occ_iou": occ_iou,
        "pred_occ": pred_occ,
    }


def test_intermediate_hooks_cover_major_prediction_blocks(instrumented_debug_forward):
    capture = instrumented_debug_forward["capture"]
    expected_names = {
        "spatial.stem",
        "spatial.block1",
        "spatial.block2",
        "spatial.out_proj",
        "temporal.gru",
        "horizon.mlp",
        "occupancy.refine",
        "occupancy.classifier",
        "trajectory.agent_proj",
        "trajectory.delta_mlp",
    }
    missing = sorted(expected_names - set(capture.keys()))
    assert not missing, f"Missing intermediate captures: {missing}"
    for name in expected_names:
        assert capture[name], f"Hook for '{name}' captured no values."


def test_shape_assertions_at_critical_boundaries(instrumented_debug_forward):
    data = instrumented_debug_forward
    cfg = data["cfg"]
    capture = data["capture"]
    outputs = data["outputs"]
    decoded = data["decoded"]
    history_bev = data["history_bev"]

    batch_size = history_bev.shape[0]
    history_steps = history_bev.shape[1]
    bev_h, bev_w = cfg.bev_hw
    mid_channels = max(cfg.embed_dims // 2, 16)
    stage1_h = _conv2d_out(bev_h, kernel=3, stride=2, padding=1)
    stage1_w = _conv2d_out(bev_w, kernel=3, stride=2, padding=1)
    stage2_h = _conv2d_out(stage1_h, kernel=3, stride=2, padding=1)
    stage2_w = _conv2d_out(stage1_w, kernel=3, stride=2, padding=1)

    assert _first_tensor(capture["spatial.stem"][-1]).shape == (
        batch_size * history_steps,
        mid_channels,
        bev_h,
        bev_w,
    )
    assert _first_tensor(capture["spatial.block1"][-1]).shape == (
        batch_size * history_steps,
        mid_channels,
        stage1_h,
        stage1_w,
    )
    assert _first_tensor(capture["spatial.block2"][-1]).shape == (
        batch_size * history_steps,
        cfg.embed_dims,
        stage2_h,
        stage2_w,
    )
    assert _first_tensor(capture["spatial.out_proj"][-1]).shape == (
        batch_size * history_steps,
        cfg.embed_dims,
        stage2_h,
        stage2_w,
    )
    assert _first_tensor(capture["temporal.gru"][-1]).shape == (
        batch_size,
        cfg.history_steps,
        cfg.embed_dims,
    )
    assert _first_tensor(capture["horizon.mlp"][-1]).shape == (
        batch_size,
        cfg.future_steps,
        cfg.embed_dims,
    )
    assert _first_tensor(capture["occupancy.refine"][-1]).shape == (
        batch_size * cfg.future_steps,
        cfg.embed_dims,
        stage2_h,
        stage2_w,
    )
    assert _first_tensor(capture["occupancy.classifier"][-1]).shape == (
        batch_size * cfg.future_steps,
        cfg.occupancy_classes * cfg.depth_bins,
        stage2_h,
        stage2_w,
    )
    assert _first_tensor(capture["trajectory.agent_proj"][-1]).shape == (
        batch_size,
        cfg.num_agents,
        cfg.embed_dims,
    )
    assert _first_tensor(capture["trajectory.delta_mlp"][-1]).shape == (
        batch_size,
        cfg.num_agents,
        cfg.future_steps,
        2,
    )

    assert outputs["occupancy_logits"].shape == (
        batch_size,
        cfg.future_steps,
        cfg.occupancy_classes,
        cfg.depth_bins,
        stage2_h,
        stage2_w,
    )
    assert outputs["trajectory"].shape == (batch_size, cfg.num_agents, cfg.future_steps, 2)
    assert outputs["velocity"].shape == (batch_size, cfg.num_agents, cfg.future_steps, 2)
    assert outputs["delta_velocity"].shape == (batch_size, cfg.num_agents, cfg.future_steps, 2)
    assert outputs["horizon_tokens"].shape == (batch_size, cfg.future_steps, cfg.embed_dims)
    assert outputs["temporal_sequence"].shape == (batch_size, cfg.history_steps, cfg.embed_dims)

    assert len(decoded) == batch_size
    for sample in decoded:
        assert sample["occupancy"].shape == (cfg.future_steps, cfg.depth_bins, stage2_h, stage2_w)
        assert sample["occupancy"].dtype == torch.bool
        assert sample["trajectory"].shape == (cfg.num_agents, cfg.future_steps, 2)
        assert sample["velocity"].shape == (cfg.num_agents, cfg.future_steps, 2)


def test_intermediate_and_final_tensors_are_finite(instrumented_debug_forward):
    data = instrumented_debug_forward
    capture = data["capture"]
    outputs = data["outputs"]
    decoded = data["decoded"]

    for name, values in capture.items():
        assert values, f"No values captured for '{name}'."
        for value in values:
            tensors = list(_iter_tensors(value))
            assert tensors, f"No tensor found in captured output for '{name}'."
            for tensor in tensors:
                assert torch.isfinite(tensor).all(), f"Non-finite values found in intermediate '{name}'."

    for name, value in outputs.items():
        tensors = list(_iter_tensors(value))
        assert tensors, f"No tensor found in output '{name}'."
        for tensor in tensors:
            assert torch.isfinite(tensor).all(), f"Non-finite values found in final output '{name}'."

    for sample in decoded:
        for name, value in sample.items():
            tensors = list(_iter_tensors(value))
            assert tensors, f"No tensor found in decoded output '{name}'."
            for tensor in tensors:
                assert torch.isfinite(tensor).all(), f"Non-finite values found in decoded output '{name}'."


def test_prediction_task_specific_integrity_checks(instrumented_debug_forward):
    data = instrumented_debug_forward
    cfg = data["cfg"]
    outputs = data["outputs"]
    metric_smoke = data["metric_smoke"]
    occ_iou = data["occ_iou"]

    # Horizon/time-axis integrity.
    assert outputs["occupancy_logits"].shape[1] == cfg.future_steps
    assert outputs["trajectory"].shape[2] == cfg.future_steps
    horizon_deltas = outputs["horizon_tokens"][:, 1:] - outputs["horizon_tokens"][:, :-1]
    assert horizon_deltas.abs().sum() > 0, "All horizon tokens are identical."

    # Trajectory consistency with velocity integration.
    consistency_error = trajectory_consistency_error(
        outputs["trajectory"],
        outputs["velocity"],
        dt=cfg.dt,
    )
    assert consistency_error.item() < 1e-5, f"Trajectory consistency error too high: {consistency_error.item()}"

    # Metric smoke checks.
    assert metric_smoke["ade"] >= 0.0
    assert metric_smoke["fde"] >= 0.0
    assert torch.isfinite(torch.tensor(metric_smoke["ade"]))
    assert torch.isfinite(torch.tensor(metric_smoke["fde"]))
    assert 0.0 <= occ_iou <= 1.0


def test_time_index_contract_validation_and_monotonicity():
    cfg = debug_forward_config(
        history_steps=4,
        future_steps=5,
        num_agents=6,
        in_channels=16,
        embed_dims=64,
        bev_hw=(24, 24),
        depth_bins=4,
    )
    model = SurroundOccPredictionLite(cfg).eval()
    history_bev = torch.randn(1, cfg.history_steps, cfg.in_channels, cfg.bev_hw[0], cfg.bev_hw[1])
    agent_states = torch.randn(1, cfg.num_agents, 4)
    history_time = torch.tensor([0.0, 0.7, 1.6, 2.8], dtype=history_bev.dtype)
    future_time = torch.tensor([0.4, 1.0, 1.9, 3.3, 5.2], dtype=history_bev.dtype)

    with torch.no_grad():
        outputs = model(
            history_bev,
            agent_states,
            history_time_indices=history_time,
            future_time_indices=future_time,
            decode=False,
        )
    assert outputs["future_time_indices"].shape == (1, cfg.future_steps)
    assert torch.allclose(outputs["future_time_indices"][0], future_time, atol=1e-6, rtol=1e-6)
    assert bool(torch.all(outputs["future_time_indices"][:, 1:] > outputs["future_time_indices"][:, :-1]))

    with pytest.raises(ValueError, match="strictly increasing"):
        model(
            history_bev,
            agent_states,
            history_time_indices=torch.tensor([0.0, 1.0, 1.0, 2.0], dtype=history_bev.dtype),
            decode=False,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        model(
            history_bev,
            agent_states,
            future_time_indices=torch.tensor([0.4, 1.0, 0.9, 1.8, 3.0], dtype=history_bev.dtype),
            decode=False,
        )


def test_decode_parity_semantics_for_occupancy_and_trajectory(instrumented_debug_forward):
    cfg = instrumented_debug_forward["cfg"]
    outputs = instrumented_debug_forward["outputs"]
    decoded = instrumented_debug_forward["decoded"]

    occupancy_probs = torch.softmax(outputs["occupancy_logits"], dim=2)
    expected_occupied_prob = 1.0 - occupancy_probs[:, :, 0]
    expected_occupancy = expected_occupied_prob > 0.5
    expected_semantic = occupancy_probs.argmax(dim=2) if cfg.occupancy_classes > 1 else None

    for batch_idx, sample in enumerate(decoded):
        expected_keys = {
            "occupancy",
            "occupancy_prob",
            "trajectory",
            "velocity",
            "future_time_indices",
        }
        if cfg.occupancy_classes > 1:
            expected_keys.add("occupancy_semantic")
        assert set(sample.keys()) == expected_keys
        assert sample["occupancy"].dtype == torch.bool
        assert sample["occupancy"].shape[0] == cfg.future_steps
        assert sample["occupancy"].shape[1] == cfg.depth_bins
        assert torch.equal(sample["occupancy"], expected_occupancy[batch_idx])
        assert torch.allclose(sample["occupancy_prob"], expected_occupied_prob[batch_idx], atol=1e-6, rtol=1e-6)
        assert torch.allclose(sample["trajectory"], outputs["trajectory"][batch_idx], atol=1e-6, rtol=1e-6)
        assert torch.allclose(sample["velocity"], outputs["velocity"][batch_idx], atol=1e-6, rtol=1e-6)
        assert torch.allclose(
            sample["future_time_indices"],
            outputs["future_time_indices"][batch_idx],
            atol=1e-6,
            rtol=1e-6,
        )
        assert bool(torch.all(sample["future_time_indices"][1:] > sample["future_time_indices"][:-1]))
        if expected_semantic is not None:
            assert torch.equal(sample["occupancy_semantic"], expected_semantic[batch_idx])


def test_future_time_deltas_control_trajectory_integration():
    cfg = debug_forward_config(
        history_steps=4,
        future_steps=5,
        num_agents=6,
        in_channels=16,
        embed_dims=64,
        bev_hw=(24, 24),
        depth_bins=4,
    )
    model = SurroundOccPredictionLite(cfg).eval()
    history_bev = torch.randn(2, cfg.history_steps, cfg.in_channels, cfg.bev_hw[0], cfg.bev_hw[1])
    agent_states = torch.randn(2, cfg.num_agents, 4)
    future_time = torch.tensor([0.3, 1.1, 2.4, 4.2, 6.5], dtype=history_bev.dtype)

    with torch.no_grad():
        outputs = model(
            history_bev,
            agent_states,
            future_time_indices=future_time,
            decode=False,
        )

    time_deltas = torch.cat((future_time[:1], future_time[1:] - future_time[:-1]), dim=0)
    expected_trajectory = agent_states[..., :2].unsqueeze(2) + torch.cumsum(
        outputs["velocity"] * time_deltas.view(1, 1, cfg.future_steps, 1),
        dim=2,
    )
    assert torch.allclose(outputs["trajectory"], expected_trajectory, atol=1e-5, rtol=1e-5)

    consistency_error = trajectory_consistency_error(
        outputs["trajectory"],
        outputs["velocity"],
        dt=time_deltas,
    )
    assert consistency_error.item() < 1e-5
