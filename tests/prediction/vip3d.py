"""Intermediate tensor and prediction-contract tests for VIP3D-lite."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from pytorch_implementation.prediction.vip3d.config import debug_forward_config
from pytorch_implementation.prediction.vip3d.model import VIP3DLite, compute_ade_fde


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


def _build_debug_batch(cfg):
    torch.manual_seed(13)
    batch_size = 2
    num_agents = 5

    base_xy = torch.randn(batch_size, num_agents, 2)
    velocity = torch.randn(batch_size, num_agents, 2) * 0.2
    history_time = torch.arange(cfg.history_steps, dtype=torch.float32).view(1, 1, cfg.history_steps, 1)
    future_time = torch.arange(1, cfg.future_steps + 1, dtype=torch.float32).view(1, 1, cfg.future_steps, 1)

    history_xy = base_xy.unsqueeze(2) + history_time * velocity.unsqueeze(2)
    history_vel = velocity.unsqueeze(2).expand(-1, -1, cfg.history_steps, -1)
    agent_history = torch.cat([history_xy, history_vel], dim=-1)

    agent_valid = torch.ones(batch_size, num_agents, cfg.history_steps, dtype=torch.bool)
    agent_valid[0, 0, -1] = False
    agent_valid[1, 3, 0] = False

    map_polylines = torch.randn(
        batch_size,
        cfg.map_tokens,
        cfg.map_points_per_token,
        cfg.map_input_dim,
    )

    last_observed = history_xy[:, :, -1]
    gt_future = last_observed.unsqueeze(2) + future_time * velocity.unsqueeze(2)
    gt_valid = torch.ones(batch_size, num_agents, cfg.future_steps, dtype=torch.bool)
    gt_valid[0, 2, -2:] = False

    return {
        "batch_size": batch_size,
        "num_agents": num_agents,
        "agent_history": agent_history,
        "agent_valid": agent_valid,
        "map_polylines": map_polylines,
        "gt_future": gt_future,
        "gt_valid": gt_valid,
    }


@pytest.fixture()
def instrumented_debug_forward():
    cfg = debug_forward_config()
    model = VIP3DLite(cfg).eval()
    batch = _build_debug_batch(cfg)

    capture: dict[str, Any] = {}
    handles = []
    _register_hook(model.history_input_proj, "history.input_proj", capture, handles)
    _register_hook(model.history_encoder.layers[0], "history.encoder.layer0", capture, handles)
    _register_hook(model.history_norm, "history.norm", capture, handles)
    _register_hook(model.map_point_mlp, "map.point_mlp", capture, handles)
    _register_hook(model.map_token_proj, "map.token_proj", capture, handles)
    _register_hook(model.agent_map_attention, "fusion.cross_attention", capture, handles)
    _register_hook(model.fusion_norm, "fusion.norm", capture, handles)
    _register_hook(model.decoder.trunk, "decoder.trunk", capture, handles)
    _register_hook(model.decoder.mode_head, "decoder.mode_head", capture, handles)
    _register_hook(model.decoder.delta_head, "decoder.delta_head", capture, handles)

    with torch.no_grad():
        outputs = model(
            agent_history=batch["agent_history"],
            map_polylines=batch["map_polylines"],
            agent_valid=batch["agent_valid"],
        )

    for handle in handles:
        handle.remove()

    return {"cfg": cfg, "batch": batch, "capture": capture, "outputs": outputs}


def test_intermediate_hooks_cover_major_prediction_stages(instrumented_debug_forward):
    capture = instrumented_debug_forward["capture"]
    expected_names = {
        "history.input_proj",
        "history.encoder.layer0",
        "history.norm",
        "map.point_mlp",
        "map.token_proj",
        "fusion.cross_attention",
        "fusion.norm",
        "decoder.trunk",
        "decoder.mode_head",
        "decoder.delta_head",
    }
    missing = sorted(expected_names - set(capture.keys()))
    assert not missing, f"Missing intermediate captures: {missing}"


def test_prediction_horizon_and_trajectory_contract(instrumented_debug_forward):
    cfg = instrumented_debug_forward["cfg"]
    batch = instrumented_debug_forward["batch"]
    outputs = instrumented_debug_forward["outputs"]

    batch_size = batch["batch_size"]
    num_agents = batch["num_agents"]

    assert outputs["history_tokens"].shape == (batch_size, num_agents, cfg.history_steps, cfg.hidden_dim)
    assert outputs["map_tokens"].shape == (batch_size, cfg.map_tokens, cfg.hidden_dim)
    assert outputs["fused_tokens"].shape == (batch_size, num_agents, cfg.hidden_dim)
    assert outputs["mode_logits"].shape == (batch_size, num_agents, cfg.num_modes)
    assert outputs["mode_probs"].shape == (batch_size, num_agents, cfg.num_modes)
    assert outputs["traj_deltas"].shape == (batch_size, num_agents, cfg.num_modes, cfg.future_steps, 2)
    assert outputs["trajectories"].shape == (batch_size, num_agents, cfg.num_modes, cfg.future_steps, 2)
    assert outputs["best_mode"].shape == (batch_size, num_agents)
    assert outputs["best_trajectory"].shape == (batch_size, num_agents, cfg.future_steps, 2)

    # Prediction-specific contract: each agent horizon is exactly T_f and mode probs are normalized.
    assert torch.allclose(outputs["mode_probs"].sum(dim=-1), torch.ones(batch_size, num_agents), atol=1e-5)

    # Trajectory consistency: finite incremental motion equals cumulative-delta construction.
    trajectory_step_diff = outputs["trajectories"][:, :, :, 1:, :] - outputs["trajectories"][:, :, :, :-1, :]
    assert torch.allclose(trajectory_step_diff, outputs["traj_deltas"][:, :, :, 1:, :], atol=1e-5)

    gather_idx = outputs["best_mode"].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
        batch_size, num_agents, 1, cfg.future_steps, 2
    )
    expected_best = outputs["trajectories"].gather(dim=2, index=gather_idx).squeeze(2)
    assert torch.allclose(outputs["best_trajectory"], expected_best, atol=1e-5)


def test_finite_values_and_metric_smoke(instrumented_debug_forward):
    batch = instrumented_debug_forward["batch"]
    capture = instrumented_debug_forward["capture"]
    outputs = instrumented_debug_forward["outputs"]

    for name, value in capture.items():
        tensors = list(_iter_tensors(value))
        assert tensors, f"No tensor found in captured output for '{name}'."
        for tensor in tensors:
            assert torch.isfinite(tensor).all(), f"Non-finite values found in intermediate '{name}'."

    for name, value in outputs.items():
        tensors = list(_iter_tensors(value))
        assert tensors, f"No tensor found in output '{name}'."
        for tensor in tensors:
            assert torch.isfinite(tensor).all(), f"Non-finite values found in output '{name}'."

    metrics = compute_ade_fde(
        trajectories=outputs["trajectories"],
        gt_future=batch["gt_future"],
        gt_valid=batch["gt_valid"],
    )

    # Prediction-specific metric smoke test: ADE/FDE must be finite non-negative scalars.
    assert metrics["ade"].ndim == 0
    assert metrics["fde"].ndim == 0
    assert torch.isfinite(metrics["ade"])
    assert torch.isfinite(metrics["fde"])
    assert metrics["ade"] >= 0
    assert metrics["fde"] >= 0
