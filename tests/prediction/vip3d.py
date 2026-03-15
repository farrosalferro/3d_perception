"""Intermediate tensor and prediction-contract tests for VIP3D-lite."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from pytorch_implementation.prediction.vip3d.config import debug_forward_config
from pytorch_implementation.prediction.vip3d.model import VIP3DLite, compute_ade_fde
from tests._shared.hook_helpers import register_hook_overwrite
from tests._shared.tensor_helpers import iter_tensors


def _iter_tensors(value: Any):
    yield from iter_tensors(value)


def _register_hook(module, name: str, capture: dict[str, Any], handles: list) -> None:
    register_hook_overwrite(module, name, capture, handles)


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


def _build_metadata(
    cfg,
    *,
    batch_size: int,
    history_start: int = 0,
    timestamp: float | None = None,
    frame_index: int | None = None,
) -> dict[str, torch.Tensor]:
    metadata: dict[str, torch.Tensor] = {
        "history_time_indices": torch.arange(
            history_start,
            history_start + cfg.history_steps,
            dtype=torch.long,
        ),
        "l2g_r_mat": torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).clone(),
        "l2g_t": torch.zeros(batch_size, 3),
    }
    if timestamp is not None:
        metadata["timestamp"] = torch.full((batch_size,), float(timestamp))
    if frame_index is not None:
        metadata["frame_index"] = torch.full((batch_size,), int(frame_index), dtype=torch.long)
    return metadata


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

    return {"cfg": cfg, "batch": batch, "capture": capture, "outputs": outputs, "model": model}


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


def test_metadata_contract_validation_and_runtime_requirements(instrumented_debug_forward):
    cfg = instrumented_debug_forward["cfg"]
    batch = instrumented_debug_forward["batch"]
    model = instrumented_debug_forward["model"]

    with pytest.raises(ValueError, match="strictly increasing"):
        model(
            agent_history=batch["agent_history"],
            map_polylines=batch["map_polylines"],
            agent_valid=batch["agent_valid"],
            metadata={"history_time_indices": [0, 2, 1, 3]},
            update_memory=False,
        )

    metadata = _build_metadata(
        cfg,
        batch_size=batch["batch_size"],
        history_start=7,
        timestamp=1.0,
        frame_index=5,
    )
    with torch.no_grad():
        outputs = model(
            agent_history=batch["agent_history"],
            map_polylines=batch["map_polylines"],
            agent_valid=batch["agent_valid"],
            metadata=metadata,
            update_memory=False,
        )

    expected_history_indices = torch.arange(7, 7 + cfg.history_steps, device=outputs["history_time_indices"].device)
    expected_history_indices = expected_history_indices.unsqueeze(0).expand(batch["batch_size"], -1)
    assert torch.equal(outputs["history_time_indices"], expected_history_indices)

    with pytest.raises(ValueError, match="metadata must include 'timestamp' or 'frame_index'"):
        model(
            agent_history=batch["agent_history"],
            map_polylines=batch["map_polylines"],
            agent_valid=batch["agent_valid"],
            metadata={"history_time_indices": [7, 8, 9, 10]},
            runtime_state=outputs["runtime_state"],
            update_memory=False,
        )


def test_decode_parity_semantics_for_tracks_and_modes(instrumented_debug_forward):
    cfg = instrumented_debug_forward["cfg"]
    batch = instrumented_debug_forward["batch"]
    outputs = instrumented_debug_forward["outputs"]

    assert torch.allclose(outputs["mode_log_probs"].exp(), outputs["mode_probs"], atol=1e-6, rtol=1e-6)
    assert torch.equal(outputs["best_mode"], outputs["mode_probs"].argmax(dim=-1))
    assert torch.allclose(outputs["pred_outputs"], outputs["trajectories"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(outputs["pred_probs"], outputs["mode_probs"], atol=1e-6, rtol=1e-6)

    decoded_tracks = outputs["decoded_tracks"]
    assert len(decoded_tracks) == batch["batch_size"]
    for batch_idx, decoded in enumerate(decoded_tracks):
        expected_keys = {
            "bboxes",
            "scores",
            "labels",
            "track_scores",
            "obj_idxes",
            "output_embedding",
        }
        assert set(decoded.keys()) == expected_keys
        count = decoded["scores"].shape[0]
        assert decoded["track_scores"].shape == (count,)
        assert decoded["labels"].shape == (count,)
        assert decoded["obj_idxes"].shape == (count,)
        assert decoded["output_embedding"].shape[0] == count
        assert torch.allclose(decoded["scores"], decoded["track_scores"], atol=1e-6, rtol=1e-6)
        if count > 1:
            assert bool((decoded["scores"][:-1] >= decoded["scores"][1:]).all())
        if count == 0:
            continue

        obj_idxes = decoded["obj_idxes"].to(dtype=torch.long)
        assert bool(((obj_idxes >= 0) & (obj_idxes < batch["num_agents"])).all())
        expected_embeddings = outputs["memory_tokens"][batch_idx].index_select(0, obj_idxes)
        assert torch.allclose(decoded["output_embedding"], expected_embeddings, atol=1e-6, rtol=1e-6)
        assert bool(((decoded["labels"] >= 0) & (decoded["labels"] < cfg.num_modes)).all())


def test_runtime_state_temporal_update_and_gap_reset(instrumented_debug_forward):
    cfg = instrumented_debug_forward["cfg"]
    batch = instrumented_debug_forward["batch"]
    model = instrumented_debug_forward["model"]
    batch_size = batch["batch_size"]

    metadata_t0 = _build_metadata(
        cfg,
        batch_size=batch_size,
        history_start=0,
        timestamp=10.0,
        frame_index=100,
    )
    metadata_t1 = _build_metadata(
        cfg,
        batch_size=batch_size,
        history_start=1,
        timestamp=10.5,
        frame_index=101,
    )
    metadata_t2 = _build_metadata(
        cfg,
        batch_size=batch_size,
        history_start=2,
        timestamp=10.5 + cfg.metadata_time_gap_reset + 1.0,
        frame_index=102,
    )

    with torch.no_grad():
        out_t0 = model(
            agent_history=batch["agent_history"],
            map_polylines=batch["map_polylines"],
            agent_valid=batch["agent_valid"],
            metadata=metadata_t0,
            update_memory=True,
        )
        out_t1 = model(
            agent_history=batch["agent_history"],
            map_polylines=batch["map_polylines"],
            agent_valid=batch["agent_valid"],
            metadata=metadata_t1,
            runtime_state=out_t0["runtime_state"],
            update_memory=True,
        )
        out_t2 = model(
            agent_history=batch["agent_history"],
            map_polylines=batch["map_polylines"],
            agent_valid=batch["agent_valid"],
            metadata=metadata_t2,
            runtime_state=out_t1["runtime_state"],
            update_memory=False,
        )

    expected_delta = torch.full_like(out_t1["time_delta"], 0.5)
    assert torch.allclose(out_t1["time_delta"], expected_delta, atol=1e-6, rtol=1e-6)
    expected_frame_delta = torch.ones(batch_size, dtype=torch.long, device=out_t1["frame_delta"].device)
    assert torch.equal(out_t1["frame_delta"], expected_frame_delta)
    assert not bool(out_t1["temporal_reset_mask"].any())

    assert bool(out_t2["temporal_reset_mask"].all())
    assert bool(out_t2["runtime_state"]["mem_padding_mask"].all())
    assert torch.allclose(
        out_t2["runtime_state"]["mem_bank"],
        torch.zeros_like(out_t2["runtime_state"]["mem_bank"]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        out_t2["runtime_state"]["save_period"],
        torch.zeros_like(out_t2["runtime_state"]["save_period"]),
        atol=1e-6,
        rtol=1e-6,
    )


def test_runtime_state_rejects_decreasing_timestamp_and_frame_index(instrumented_debug_forward):
    cfg = instrumented_debug_forward["cfg"]
    batch = instrumented_debug_forward["batch"]
    model = instrumented_debug_forward["model"]
    metadata_t0 = _build_metadata(
        cfg,
        batch_size=batch["batch_size"],
        history_start=0,
        timestamp=3.0,
        frame_index=20,
    )

    with torch.no_grad():
        out_t0 = model(
            agent_history=batch["agent_history"],
            map_polylines=batch["map_polylines"],
            agent_valid=batch["agent_valid"],
            metadata=metadata_t0,
            update_memory=False,
        )

    with pytest.raises(ValueError, match="timestamp must be non-decreasing"):
        model(
            agent_history=batch["agent_history"],
            map_polylines=batch["map_polylines"],
            agent_valid=batch["agent_valid"],
            metadata=_build_metadata(
                cfg,
                batch_size=batch["batch_size"],
                history_start=1,
                timestamp=2.5,
                frame_index=21,
            ),
            runtime_state=out_t0["runtime_state"],
            update_memory=False,
        )
    with pytest.raises(ValueError, match="frame_index must be non-decreasing"):
        model(
            agent_history=batch["agent_history"],
            map_polylines=batch["map_polylines"],
            agent_valid=batch["agent_valid"],
            metadata=_build_metadata(
                cfg,
                batch_size=batch["batch_size"],
                history_start=1,
                frame_index=19,
            ),
            runtime_state=out_t0["runtime_state"],
            update_memory=False,
        )


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
