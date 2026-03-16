"""Intermediate tensor validation tests for planning/vad."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from pytorch_implementation.planning.common import build_debug_batch
from pytorch_implementation.planning.vad import VADLite, debug_forward_config
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
    model = VADLite(cfg).eval()
    batch = build_debug_batch(cfg.e2e, batch_size=2)

    capture: dict[str, Any] = {}
    handles = []
    _register_hook(model.vector_encoder.ego_proj, "vector.ego_proj", capture, handles)
    _register_hook(model.vector_encoder.agent_proj, "vector.agent_proj", capture, handles)
    _register_hook(model.vector_encoder.map_point_proj, "vector.map_point_proj", capture, handles)
    _register_hook(model.planner_core.cross_attn, "planner.cross_attn", capture, handles)
    _register_hook(model.planner_core.traj_head, "planner.traj_head", capture, handles)
    _register_hook(model.planner_core.score_head, "planner.score_head", capture, handles)

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
    capture = instrumented_debug_forward["capture"]
    expected = {
        "vector.ego_proj",
        "vector.agent_proj",
        "vector.map_point_proj",
        "planner.cross_attn",
        "planner.traj_head",
        "planner.score_head",
    }
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

    assert _first_tensor(capture["vector.ego_proj"]).shape == (bsz, c)
    assert _first_tensor(capture["vector.agent_proj"]).shape == (bsz, cfg.e2e.num_agents, c)
    assert _first_tensor(capture["vector.map_point_proj"]).shape == (
        bsz,
        cfg.e2e.map_polylines,
        cfg.e2e.points_per_polyline,
        c,
    )
    assert _first_tensor(capture["planner.cross_attn"]).shape == (bsz, 1, c)
    assert _first_tensor(capture["planner.traj_head"]).shape == (bsz, k * t * 2)
    assert _first_tensor(capture["planner.score_head"]).shape == (bsz, k)

    assert outputs["candidate_trajectories"].shape == (bsz, k, t, 2)
    assert outputs["candidate_scores"].shape == (bsz, k)
    assert outputs["selected_trajectory"].shape == (bsz, t, 2)
    assert outputs["feasible_mask"].shape == (bsz, k)
    assert outputs["collision_free_mask"].shape == (bsz, k)
    assert outputs["lane_distance"].shape == (bsz, k)
    assert outputs["lane_alignment"].shape == (bsz, k)
    assert outputs["constraint_cost"].shape == (bsz, k)
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

    lane_tolerance = cfg.e2e.safety_margin * 3.0
    assert torch.all(outputs["lane_distance"] >= 0)
    lane_mask = outputs["lane_distance"] <= lane_tolerance
    lane_align_mask = outputs["lane_alignment"] >= 0.0
    composite_feasible = outputs["feasible_mask"]
    assert torch.all(composite_feasible <= lane_mask)
    assert torch.all(composite_feasible <= lane_align_mask)

    gather_idx = outputs["selected_index"].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
        -1, 1, cfg.e2e.future_steps, 2
    )
    selected_expected = outputs["candidate_trajectories"].gather(dim=1, index=gather_idx).squeeze(1)
    assert torch.allclose(outputs["selected_trajectory"], selected_expected, atol=1e-6, rtol=1e-6)
