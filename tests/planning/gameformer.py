"""Intermediate tensor validation tests for planning/gameformer."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from pytorch_implementation.planning.common import build_debug_batch
from pytorch_implementation.planning.gameformer import GameFormerLite, debug_forward_config
from tests._shared.hook_helpers import register_hook_overwrite
from tests._shared.tensor_helpers import first_tensor, iter_tensors


def _first_tensor(value: Any) -> torch.Tensor | None:
    return first_tensor(value)


def _iter_tensors(value: Any):
    yield from iter_tensors(value)


def _register_hook(module, name: str, capture: dict[str, Any], handles: list) -> None:
    register_hook_overwrite(module, name, capture, handles)


@pytest.fixture()
def instrumented_debug_forward():
    cfg = debug_forward_config()
    model = GameFormerLite(cfg).eval()
    batch = build_debug_batch(cfg.e2e, batch_size=2)

    capture: dict[str, Any] = {}
    handles = []
    _register_hook(model.scene_encoder.ego_proj, "scene.ego_proj", capture, handles)
    _register_hook(model.scene_encoder.ego_gru, "scene.ego_gru", capture, handles)
    _register_hook(model.scene_encoder.agent_proj, "scene.agent_proj", capture, handles)
    _register_hook(model.scene_encoder.map_point_proj, "scene.map_point_proj", capture, handles)
    for idx, layer in enumerate(model.interaction_layers):
        _register_hook(layer.ego_attn, f"interaction.{idx}.ego_attn", capture, handles)
        _register_hook(layer.agent_attn, f"interaction.{idx}.agent_attn", capture, handles)
    _register_hook(model.candidate_delta_head, "head.candidate_delta", capture, handles)
    _register_hook(model.candidate_score_head, "head.candidate_score", capture, handles)
    _register_hook(model.agent_delta_head, "head.agent_delta", capture, handles)

    with torch.no_grad():
        outputs = model(
            ego_history=batch.ego_history,
            agent_states=batch.agent_states,
            map_polylines=batch.map_polylines,
            route_features=batch.route_features,
        )

    for handle in handles:
        handle.remove()

    return {"cfg": cfg, "model": model, "batch": batch, "capture": capture, "outputs": outputs}


def test_intermediate_hooks_cover_major_blocks(instrumented_debug_forward):
    cfg = instrumented_debug_forward["cfg"]
    capture = instrumented_debug_forward["capture"]
    expected = {
        "scene.ego_proj",
        "scene.ego_gru",
        "scene.agent_proj",
        "scene.map_point_proj",
        "head.candidate_delta",
        "head.candidate_score",
        "head.agent_delta",
    }
    for idx in range(cfg.game_levels):
        expected.add(f"interaction.{idx}.ego_attn")
        expected.add(f"interaction.{idx}.agent_attn")
    missing = sorted(expected - set(capture.keys()))
    assert not missing, f"Missing intermediate captures: {missing}"


def test_shape_contracts(instrumented_debug_forward):
    cfg = instrumented_debug_forward["cfg"]
    batch = instrumented_debug_forward["batch"]
    capture = instrumented_debug_forward["capture"]
    outputs = instrumented_debug_forward["outputs"]

    bsz = batch.ego_history.shape[0]
    k = cfg.e2e.num_candidates
    t = cfg.e2e.future_steps
    c = cfg.e2e.hidden_dim

    assert _first_tensor(capture["scene.ego_proj"]).shape == (bsz, cfg.e2e.history_steps, c)
    assert _first_tensor(capture["scene.agent_proj"]).shape == (bsz, cfg.e2e.num_agents, c)
    assert _first_tensor(capture["scene.map_point_proj"]).shape == (
        bsz,
        cfg.e2e.map_polylines,
        cfg.e2e.points_per_polyline,
        c,
    )
    for idx in range(cfg.game_levels):
        assert _first_tensor(capture[f"interaction.{idx}.ego_attn"]).shape == (bsz, 1, c)
        assert _first_tensor(capture[f"interaction.{idx}.agent_attn"]).shape == (bsz, cfg.e2e.num_agents, c)
    assert _first_tensor(capture["head.candidate_delta"]).shape == (bsz, k * t * 2)
    assert _first_tensor(capture["head.candidate_score"]).shape == (bsz, k)
    assert _first_tensor(capture["head.agent_delta"]).shape == (bsz, cfg.e2e.num_agents, t * 2)

    assert outputs["candidate_trajectories"].shape == (bsz, k, t, 2)
    assert outputs["candidate_scores"].shape == (bsz, k)
    assert outputs["selected_trajectory"].shape == (bsz, t, 2)
    assert outputs["agent_future"].shape == (bsz, cfg.e2e.num_agents, t, 2)
    assert outputs["level_ego_tokens"].shape == (bsz, cfg.game_levels, c)
    assert outputs["level_agent_tokens"].shape == (bsz, cfg.game_levels, cfg.e2e.num_agents, c)
    assert outputs["time_stamps"].shape == (t,)


def test_finite_values(instrumented_debug_forward):
    capture = instrumented_debug_forward["capture"]
    outputs = instrumented_debug_forward["outputs"]

    for name, value in capture.items():
        tensors = list(_iter_tensors(value))
        assert tensors, f"No tensor found in captured output for '{name}'."
        for tensor in tensors:
            assert torch.isfinite(tensor).all(), f"Non-finite values found in intermediate '{name}'."

    for name, value in outputs.items():
        tensors = list(_iter_tensors(value))
        if not tensors:
            continue
        for tensor in tensors:
            assert torch.isfinite(tensor).all(), f"Non-finite values found in output '{name}'."


def test_kinematic_and_safety_checks(instrumented_debug_forward):
    cfg = instrumented_debug_forward["cfg"]
    outputs = instrumented_debug_forward["outputs"]

    expected_collision = outputs["min_distance"] >= cfg.e2e.safety_margin
    assert torch.equal(outputs["collision_free_mask"], expected_collision)
    expected_violation = torch.relu(cfg.e2e.safety_margin - outputs["min_distance"])
    assert torch.allclose(outputs["safety_margin_violation"], expected_violation, atol=1e-6, rtol=1e-6)

    if outputs["feasible_mask"].any():
        speed = torch.linalg.norm(outputs["velocity"], dim=-1).amax(dim=-1)
        accel = torch.linalg.norm(outputs["acceleration"], dim=-1).amax(dim=-1)
        curvature = outputs["curvature"].amax(dim=-1)
        feasible = outputs["feasible_mask"]
        assert torch.all(speed[feasible] <= cfg.e2e.max_speed + 1e-5)
        assert torch.all(accel[feasible] <= cfg.e2e.max_accel + 1e-5)
        assert torch.all(curvature[feasible] <= cfg.e2e.max_curvature + 1e-5)

    gather_idx = outputs["selected_index"].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
        -1, 1, cfg.e2e.future_steps, 2
    )
    selected_expected = outputs["candidate_trajectories"].gather(dim=1, index=gather_idx).squeeze(1)
    assert torch.allclose(outputs["selected_trajectory"], selected_expected, atol=1e-6, rtol=1e-6)
