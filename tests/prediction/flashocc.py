"""Intermediate tensor validation tests for prediction/flashocc.

Run:
    conda activate 3d_perception
    pytest tests/prediction/flashocc.py -q
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from pytorch_implementation.prediction.flashocc.config import debug_forward_config
from pytorch_implementation.prediction.flashocc.metrics import (
    average_displacement_error,
    final_displacement_error,
    trajectory_smoothness_l2,
)
from pytorch_implementation.prediction.flashocc.model import FlashOccLite
from tests._shared.hook_helpers import register_hook_overwrite
from tests._shared.tensor_helpers import conv2d_out, first_tensor, iter_tensors


def _first_tensor(value: Any) -> torch.Tensor | None:
    return first_tensor(value)


def _iter_tensors(value: Any):
    yield from iter_tensors(value)


def _register_hook(module, name: str, capture: dict[str, Any], handles: list) -> None:
    register_hook_overwrite(module, name, capture, handles)


def _conv2d_out(size: int, kernel: int, stride: int, padding: int) -> int:
    return conv2d_out(size, kernel, stride, padding)


def _identity_history_to_key(
    batch_size: int,
    history_steps: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    eye = torch.eye(3, device=device, dtype=dtype)
    return eye.view(1, 1, 3, 3).expand(batch_size, history_steps, -1, -1).clone()


@pytest.fixture()
def instrumented_debug_forward():
    cfg = debug_forward_config(
        num_history=4,
        pred_horizon=8,
        bev_h=48,
        bev_w=64,
        num_queries=12,
        num_modes=3,
        topk=6,
    )
    model = FlashOccLite(cfg).eval()

    batch_size = 2
    occ_seq = torch.randn(
        batch_size,
        cfg.num_history,
        cfg.backbone.in_channels,
        cfg.bev_h,
        cfg.bev_w,
    )

    capture: dict[str, Any] = {}
    handles = []
    _register_hook(model.backbone.stem, "backbone.stem", capture, handles)
    for idx, block in enumerate(model.backbone.blocks):
        _register_hook(block, f"backbone.block{idx}", capture, handles)
    _register_hook(model.temporal_mixer.temporal_conv, "temporal.temporal_conv", capture, handles)
    _register_hook(model.temporal_mixer.proj, "temporal.proj", capture, handles)
    _register_hook(model.prediction_head.query_proj, "head.query_proj", capture, handles)
    _register_hook(model.prediction_head.cross_attn, "head.cross_attn", capture, handles)
    _register_hook(model.prediction_head.traj_head, "head.traj_head", capture, handles)
    _register_hook(model.prediction_head.mode_head, "head.mode_head", capture, handles)

    with torch.no_grad():
        outputs = model(occ_seq, decode=False)

    for handle in handles:
        handle.remove()

    assert isinstance(outputs, dict)
    return {
        "cfg": cfg,
        "batch_size": batch_size,
        "occ_seq": occ_seq,
        "capture": capture,
        "outputs": outputs,
        "model": model,
    }


def test_intermediate_hooks_cover_major_layers(instrumented_debug_forward):
    capture = instrumented_debug_forward["capture"]
    expected = {
        "backbone.stem",
        "backbone.block0",
        "backbone.block1",
        "temporal.temporal_conv",
        "temporal.proj",
        "head.query_proj",
        "head.cross_attn",
        "head.traj_head",
        "head.mode_head",
    }
    missing = sorted(expected - set(capture.keys()))
    assert not missing, f"Missing intermediate captures: {missing}"


def test_intermediate_shapes_match_debug_config(instrumented_debug_forward):
    data = instrumented_debug_forward
    cfg = data["cfg"]
    capture = data["capture"]
    outputs = data["outputs"]
    batch_size = data["batch_size"]

    h2 = _conv2d_out(cfg.bev_h, cfg.backbone.stem_kernel, cfg.backbone.stem_stride, cfg.backbone.stem_padding)
    w2 = _conv2d_out(cfg.bev_w, cfg.backbone.stem_kernel, cfg.backbone.stem_stride, cfg.backbone.stem_padding)
    embed = cfg.backbone.embed_dims

    assert _first_tensor(capture["backbone.stem"]).shape == (batch_size * cfg.num_history, embed, h2, w2)
    assert _first_tensor(capture["backbone.block0"]).shape == (batch_size * cfg.num_history, embed, h2, w2)
    assert _first_tensor(capture["backbone.block1"]).shape == (batch_size * cfg.num_history, embed, h2, w2)
    assert _first_tensor(capture["temporal.temporal_conv"]).shape == (batch_size * h2 * w2, embed, cfg.num_history)
    assert _first_tensor(capture["temporal.proj"]).shape == (batch_size, embed, h2, w2)
    assert _first_tensor(capture["head.query_proj"]).shape == (batch_size, embed)
    assert _first_tensor(capture["head.cross_attn"]).shape == (batch_size, cfg.num_queries, embed)
    assert _first_tensor(capture["head.traj_head"]).shape == (
        batch_size,
        cfg.num_queries,
        cfg.num_modes * cfg.pred_horizon * 2,
    )
    assert _first_tensor(capture["head.mode_head"]).shape == (batch_size, cfg.num_queries, cfg.num_modes)

    assert outputs["bev_sequence"].shape == (batch_size, cfg.num_history, embed, h2, w2)
    assert outputs["bev_fused"].shape == (batch_size, embed, h2, w2)
    assert outputs["temporal_tokens"].shape == (batch_size * h2 * w2, embed, cfg.num_history)
    assert outputs["query_tokens"].shape == (batch_size, cfg.num_queries, embed)
    assert outputs["anchor_xy"].shape == (batch_size, cfg.num_queries, 2)
    assert outputs["traj_deltas"].shape == (batch_size, cfg.num_queries, cfg.num_modes, cfg.pred_horizon, 2)
    assert outputs["traj_positions"].shape == (batch_size, cfg.num_queries, cfg.num_modes, cfg.pred_horizon, 2)
    assert outputs["traj_velocity"].shape == (batch_size, cfg.num_queries, cfg.num_modes, cfg.pred_horizon, 2)
    assert outputs["mode_logits"].shape == (batch_size, cfg.num_queries, cfg.num_modes)
    assert outputs["time_stamps"].shape == (cfg.pred_horizon,)


def test_intermediate_and_final_tensors_are_finite(instrumented_debug_forward):
    capture = instrumented_debug_forward["capture"]
    outputs = instrumented_debug_forward["outputs"]

    for name, value in capture.items():
        tensors = list(_iter_tensors(value))
        assert tensors, f"No tensor found in captured output for '{name}'."
        for tensor in tensors:
            assert torch.isfinite(tensor).all(), f"Non-finite values found in intermediate '{name}'."

    for name, value in outputs.items():
        tensors = list(_iter_tensors(value))
        assert tensors, f"No tensor found in model output '{name}'."
        for tensor in tensors:
            assert torch.isfinite(tensor).all(), f"Non-finite values found in output '{name}'."


def test_prediction_horizon_and_trajectory_contracts(instrumented_debug_forward):
    data = instrumented_debug_forward
    cfg = data["cfg"]
    outputs = data["outputs"]
    model = data["model"]
    occ_seq = data["occ_seq"]
    batch_size = data["batch_size"]

    time_stamps = outputs["time_stamps"]
    expected_time = torch.arange(1, cfg.pred_horizon + 1, dtype=time_stamps.dtype, device=time_stamps.device) * cfg.dt
    assert torch.allclose(time_stamps, expected_time)
    assert bool(torch.all(time_stamps[1:] > time_stamps[:-1]))

    reconstructed = outputs["anchor_xy"].unsqueeze(2).unsqueeze(3) + torch.cumsum(outputs["traj_deltas"], dim=3)
    assert torch.allclose(outputs["traj_positions"], reconstructed, atol=1e-6)
    assert torch.allclose(outputs["traj_velocity"] * cfg.dt, outputs["traj_deltas"], atol=1e-6)

    decoded_out = model(occ_seq, decode=True)
    assert isinstance(decoded_out, dict)
    assert "preds" in decoded_out and "decoded" in decoded_out
    decoded = decoded_out["decoded"]
    assert isinstance(decoded, list)
    assert len(decoded) == batch_size

    for sample in decoded:
        assert set(sample.keys()) == {"trajectories", "scores", "mode_indices", "query_indices"}
        k = sample["trajectories"].shape[0]
        assert k <= cfg.topk
        assert sample["trajectories"].shape == (k, cfg.pred_horizon, 2)
        assert sample["scores"].shape == (k,)
        assert sample["mode_indices"].shape == (k,)
        assert sample["query_indices"].shape == (k,)
        assert bool((sample["mode_indices"] >= 0).all() and (sample["mode_indices"] < cfg.num_modes).all())
        assert bool((sample["query_indices"] >= 0).all() and (sample["query_indices"] < cfg.num_queries).all())
        if k > 1:
            assert bool((sample["scores"][:-1] >= sample["scores"][1:]).all())


def test_decode_parity_semantics_for_trajectory_and_occupancy(instrumented_debug_forward):
    cfg = instrumented_debug_forward["cfg"]
    model = instrumented_debug_forward["model"]
    occ_seq = instrumented_debug_forward["occ_seq"]
    batch_size = instrumented_debug_forward["batch_size"]

    with torch.no_grad():
        decoded_out = model(occ_seq, decode=True)

    assert set(decoded_out.keys()) == {"preds", "decoded", "decoded_occ"}
    preds = decoded_out["preds"]
    decoded = decoded_out["decoded"]
    decoded_occ = decoded_out["decoded_occ"]
    expected_time = (
        torch.arange(1, cfg.pred_horizon + 1, device=preds["time_stamps"].device, dtype=preds["time_stamps"].dtype)
        * cfg.dt
    )
    assert torch.allclose(preds["time_stamps"], expected_time, atol=1e-6, rtol=1e-6)
    assert bool(torch.all(preds["time_stamps"][1:] > preds["time_stamps"][:-1]))
    mode_probs = preds["mode_logits"].softmax(dim=-1)
    expected_occ_labels = preds["occupancy_logits"].softmax(dim=-1).argmax(dim=-1)

    assert len(decoded) == batch_size
    assert len(decoded_occ) == batch_size
    for batch_idx, sample in enumerate(decoded):
        assert set(sample.keys()) == {"trajectories", "scores", "mode_indices", "query_indices"}
        count = sample["scores"].shape[0]
        assert sample["trajectories"].shape == (count, cfg.pred_horizon, 2)
        assert sample["mode_indices"].shape == (count,)
        assert sample["query_indices"].shape == (count,)
        if count > 1:
            assert bool((sample["scores"][:-1] >= sample["scores"][1:]).all())
        if count == 0:
            continue

        mode_indices = sample["mode_indices"].to(dtype=torch.long)
        query_indices = sample["query_indices"].to(dtype=torch.long)
        assert bool(((mode_indices >= 0) & (mode_indices < cfg.num_modes)).all())
        assert bool(((query_indices >= 0) & (query_indices < cfg.num_queries)).all())

        selected_query_traj = preds["traj_positions"][batch_idx].index_select(0, query_indices)
        expected_traj = selected_query_traj[
            torch.arange(count, device=query_indices.device),
            mode_indices,
        ]
        expected_scores = mode_probs[batch_idx, query_indices, mode_indices]
        assert torch.allclose(sample["trajectories"], expected_traj, atol=1e-6, rtol=1e-6)
        assert torch.allclose(sample["scores"], expected_scores, atol=1e-6, rtol=1e-6)

        occ_labels = decoded_occ[batch_idx]
        assert tuple(occ_labels.shape) == tuple(expected_occ_labels[batch_idx].shape)
        occ_labels = occ_labels.to(device=expected_occ_labels.device, dtype=expected_occ_labels.dtype)
        assert torch.equal(occ_labels, expected_occ_labels[batch_idx])


def test_temporal_alignment_state_update_with_history_to_key(instrumented_debug_forward):
    cfg = instrumented_debug_forward["cfg"]
    model = instrumented_debug_forward["model"]
    occ_seq = instrumented_debug_forward["occ_seq"]
    batch_size = instrumented_debug_forward["batch_size"]

    identity_history_to_key = _identity_history_to_key(
        batch_size=batch_size,
        history_steps=cfg.num_history,
        device=occ_seq.device,
        dtype=occ_seq.dtype,
    )
    shifted_history_to_key = identity_history_to_key.clone()
    shifted_history_to_key[:, 1:, 0, 2] = 1.0

    with torch.no_grad():
        outputs_identity = model(
            occ_seq,
            decode=False,
            history_to_key=identity_history_to_key,
        )
        outputs_shifted = model(
            occ_seq,
            decode=False,
            history_to_key=shifted_history_to_key,
        )

    assert torch.allclose(
        outputs_identity["bev_sequence_aligned"],
        outputs_identity["bev_sequence"],
        atol=1e-5,
        rtol=1e-5,
    )
    alignment_delta = (
        outputs_shifted["bev_sequence_aligned"][:, 1:] - outputs_shifted["bev_sequence"][:, 1:]
    ).abs().sum()
    assert float(alignment_delta) > 0.0
    assert not torch.allclose(outputs_identity["bev_fused"], outputs_shifted["bev_fused"])


def test_prediction_metric_smoke(instrumented_debug_forward):
    outputs = instrumented_debug_forward["outputs"]
    traj_positions = outputs["traj_positions"]
    mode_logits = outputs["mode_logits"]

    best_mode = mode_logits.argmax(dim=-1)
    gather_idx = best_mode.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
        -1,
        -1,
        1,
        traj_positions.shape[-2],
        traj_positions.shape[-1],
    )
    pred_best = torch.gather(traj_positions, dim=2, index=gather_idx).squeeze(2)
    gt = pred_best + 0.05 * torch.randn_like(pred_best)

    ade = average_displacement_error(pred_best, gt)
    fde = final_displacement_error(pred_best, gt)
    smooth = trajectory_smoothness_l2(pred_best)

    for name, metric in (("ADE", ade), ("FDE", fde), ("Smoothness", smooth)):
        assert metric.dim() == 0, f"{name} should be scalar."
        assert torch.isfinite(metric), f"{name} should be finite."
        assert float(metric) >= 0.0, f"{name} should be non-negative."

