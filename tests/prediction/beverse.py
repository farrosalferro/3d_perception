"""Intermediate tensor and prediction-contract tests for BEVerse-lite."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from pytorch_implementation.prediction.beverse.config import debug_forward_config
from pytorch_implementation.prediction.beverse.metrics import compute_ade_fde, select_best_mode_by_ade
from pytorch_implementation.prediction.beverse.model import BEVerseLite


def _build_dummy_img_metas(
    batch_size: int,
    *,
    seq_len: int = 1,
    include_time_indices: bool = False,
    include_future_time_stamps: bool = False,
    pred_horizon: int | None = None,
    future_dt: float = 0.5,
) -> list[dict[str, Any]]:
    metas: list[dict[str, Any]] = []
    for batch_idx in range(batch_size):
        meta: dict[str, Any] = {"sample_idx": f"sample_{batch_idx}"}
        if include_time_indices:
            start = 100 * (batch_idx + 1)
            frame_indices = list(range(start, start + seq_len))
            meta["frame_indices"] = frame_indices
            meta["timestamp_indices"] = [float(idx) * future_dt for idx in frame_indices]
        if include_future_time_stamps:
            if pred_horizon is None:
                raise ValueError("pred_horizon must be provided when include_future_time_stamps=True.")
            meta["future_time_stamps"] = [
                float(step + 1) * future_dt for step in range(int(pred_horizon))
            ]
        metas.append(meta)
    return metas


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


def _conv2d_out(size: int, kernel: int, stride: int, padding: int) -> int:
    return ((size + 2 * padding - kernel) // stride) + 1


@pytest.fixture()
def instrumented_debug_forward():
    cfg = debug_forward_config()
    model = BEVerseLite(cfg).eval()

    batch_size = 2
    height, width = 96, 160
    img = torch.randn(batch_size, cfg.num_cams, 3, height, width)
    img_metas = _build_dummy_img_metas(batch_size=batch_size)

    capture: dict[str, Any] = {}
    handles = []

    _register_hook(model.backbone_neck.backbone.stem, "backbone.stem", capture, handles)
    for idx, stage in enumerate(model.backbone_neck.backbone.stages):
        _register_hook(stage, f"backbone.stage{idx}", capture, handles)
    _register_hook(model.backbone_neck.neck.output_convs[0], "fpn.output0", capture, handles)
    _register_hook(model.bev_encoder[0], "bev_encoder.conv0", capture, handles)
    _register_hook(model.bev_encoder[3], "bev_encoder.conv1", capture, handles)
    _register_hook(model.temporal_predictor.time_embedding, "temporal.time_embedding", capture, handles)
    _register_hook(model.temporal_predictor.gru, "temporal.gru", capture, handles)
    _register_hook(model.trajectory_head.shared[1], "head.shared_fc0", capture, handles)
    _register_hook(model.trajectory_head.delta_head, "head.delta", capture, handles)
    _register_hook(model.trajectory_head.mode_head, "head.mode", capture, handles)

    with torch.no_grad():
        outputs = model(img, img_metas, decode=False)
        decoded = model(img, img_metas, decode=True)

    for handle in handles:
        handle.remove()

    assert isinstance(outputs, dict)
    return {
        "cfg": cfg,
        "batch_size": batch_size,
        "height": height,
        "width": width,
        "capture": capture,
        "outputs": outputs,
        "decoded": decoded,
    }


def test_intermediate_hooks_cover_all_major_layers(instrumented_debug_forward):
    capture = instrumented_debug_forward["capture"]
    expected_names = {
        "backbone.stem",
        "backbone.stage0",
        "backbone.stage1",
        "backbone.stage2",
        "backbone.stage3",
        "fpn.output0",
        "bev_encoder.conv0",
        "bev_encoder.conv1",
        "temporal.time_embedding",
        "temporal.gru",
        "head.shared_fc0",
        "head.delta",
        "head.mode",
    }
    missing = sorted(expected_names - set(capture.keys()))
    assert not missing, f"Missing intermediate captures: {missing}"


def test_intermediate_shapes_match_debug_config(instrumented_debug_forward):
    data = instrumented_debug_forward
    cfg = data["cfg"]
    capture = data["capture"]
    outputs = data["outputs"]
    decoded = data["decoded"]
    batch_size = data["batch_size"]
    height = data["height"]
    width = data["width"]

    cams = cfg.num_cams
    c1, c2, c3, c4 = cfg.backbone_neck.stage_channels
    stem_h = _conv2d_out(height, kernel=7, stride=2, padding=3)
    stem_w = _conv2d_out(width, kernel=7, stride=2, padding=3)
    stage1_h = _conv2d_out(stem_h, kernel=3, stride=2, padding=1)
    stage1_w = _conv2d_out(stem_w, kernel=3, stride=2, padding=1)
    stage2_h = _conv2d_out(stage1_h, kernel=3, stride=2, padding=1)
    stage2_w = _conv2d_out(stage1_w, kernel=3, stride=2, padding=1)
    stage3_h = _conv2d_out(stage2_h, kernel=3, stride=2, padding=1)
    stage3_w = _conv2d_out(stage2_w, kernel=3, stride=2, padding=1)

    assert _first_tensor(capture["backbone.stem"]).shape == (batch_size * cams, c1, stem_h, stem_w)
    assert _first_tensor(capture["backbone.stage0"]).shape == (batch_size * cams, c1, stem_h, stem_w)
    assert _first_tensor(capture["backbone.stage1"]).shape == (batch_size * cams, c2, stage1_h, stage1_w)
    assert _first_tensor(capture["backbone.stage2"]).shape == (batch_size * cams, c3, stage2_h, stage2_w)
    assert _first_tensor(capture["backbone.stage3"]).shape == (batch_size * cams, c4, stage3_h, stage3_w)
    assert _first_tensor(capture["fpn.output0"]).shape == (
        batch_size * cams,
        cfg.embed_dims,
        stage3_h,
        stage3_w,
    )

    assert _first_tensor(capture["bev_encoder.conv0"]).shape == (
        batch_size,
        cfg.embed_dims,
        cfg.bev_h,
        cfg.bev_w,
    )
    assert _first_tensor(capture["bev_encoder.conv1"]).shape == (
        batch_size,
        cfg.embed_dims,
        cfg.bev_h,
        cfg.bev_w,
    )
    assert _first_tensor(capture["temporal.time_embedding"]).shape == (
        batch_size,
        cfg.pred_horizon,
        cfg.embed_dims,
    )
    assert _first_tensor(capture["temporal.gru"]).shape == (
        batch_size,
        cfg.pred_horizon,
        cfg.embed_dims,
    )
    assert _first_tensor(capture["head.shared_fc0"]).shape == (
        batch_size,
        cfg.pred_horizon,
        cfg.embed_dims,
    )
    assert _first_tensor(capture["head.delta"]).shape == (
        batch_size,
        cfg.pred_horizon,
        cfg.num_modes * 2,
    )
    assert _first_tensor(capture["head.mode"]).shape == (batch_size, cfg.num_modes)

    assert outputs["camera_feat"].shape == (batch_size, cfg.num_cams, cfg.embed_dims, stage3_h, stage3_w)
    assert outputs["bev_embed"].shape == (batch_size, cfg.embed_dims, cfg.bev_h, cfg.bev_w)
    assert outputs["temporal_tokens"].shape == (batch_size, cfg.pred_horizon, cfg.embed_dims)
    assert outputs["trajectory_deltas"].shape == (batch_size, cfg.num_modes, cfg.pred_horizon, 2)
    assert outputs["trajectory_preds"].shape == (batch_size, cfg.num_modes, cfg.pred_horizon, 2)
    assert outputs["mode_logits"].shape == (batch_size, cfg.num_modes)
    assert outputs["mode_probs"].shape == (batch_size, cfg.num_modes)
    assert outputs["time_stamps"].shape == (cfg.pred_horizon,)

    assert isinstance(decoded, dict)
    assert "preds" in decoded
    assert "decoded" in decoded
    decoded_payload = decoded["decoded"]
    assert decoded_payload["best_mode_idx"].shape == (batch_size,)
    assert decoded_payload["best_mode_prob"].shape == (batch_size,)
    assert decoded_payload["best_trajectory"].shape == (batch_size, cfg.pred_horizon, 2)


def test_intermediate_and_final_tensors_are_finite(instrumented_debug_forward):
    capture = instrumented_debug_forward["capture"]
    outputs = instrumented_debug_forward["outputs"]
    decoded = instrumented_debug_forward["decoded"]

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

    decoded_payload = decoded["decoded"]
    for name, value in decoded_payload.items():
        tensors = list(_iter_tensors(value))
        assert tensors, f"No tensor found in decoded output '{name}'."
        for tensor in tensors:
            assert torch.isfinite(tensor).all(), f"Non-finite values found in decoded output '{name}'."


def test_prediction_horizon_and_time_axis_integrity(instrumented_debug_forward):
    cfg = instrumented_debug_forward["cfg"]
    outputs = instrumented_debug_forward["outputs"]

    time_stamps = outputs["time_stamps"]
    assert time_stamps.shape[0] == cfg.pred_horizon
    assert outputs["trajectory_preds"].shape[2] == cfg.pred_horizon
    assert outputs["trajectory_deltas"].shape[2] == cfg.pred_horizon

    expected_step = torch.full(
        (cfg.pred_horizon - 1,),
        fill_value=cfg.future_dt,
        dtype=time_stamps.dtype,
        device=time_stamps.device,
    )
    actual_step = time_stamps[1:] - time_stamps[:-1]
    assert torch.allclose(actual_step, expected_step)
    assert time_stamps[0].item() == pytest.approx(cfg.future_dt)
    assert time_stamps[-1].item() == pytest.approx(cfg.pred_horizon * cfg.future_dt)


def test_metadata_and_future_time_index_contracts_are_strict():
    cfg = debug_forward_config(pred_horizon=6)
    model = BEVerseLite(cfg).eval()

    batch_size = 1
    seq_len = 3
    img = torch.randn(batch_size, seq_len, cfg.num_cams, 3, 64, 96)
    valid_meta = _build_dummy_img_metas(
        batch_size=batch_size,
        seq_len=seq_len,
        include_time_indices=True,
        include_future_time_stamps=True,
        pred_horizon=cfg.pred_horizon,
        future_dt=cfg.future_dt,
    )

    with torch.no_grad():
        outputs = model(img, valid_meta, decode=False)

    expected_time_stamps = torch.as_tensor(
        valid_meta[0]["future_time_stamps"],
        device=outputs["time_stamps"].device,
        dtype=outputs["time_stamps"].dtype,
    )
    assert torch.allclose(outputs["time_stamps"], expected_time_stamps)
    assert bool(torch.all(outputs["time_stamps"][1:] > outputs["time_stamps"][:-1]))

    invalid_meta = _build_dummy_img_metas(
        batch_size=batch_size,
        seq_len=seq_len,
        include_time_indices=True,
    )
    invalid_meta[0]["frame_indices"] = [10, 12, 11]
    with pytest.raises(ValueError, match="strictly increasing"):
        model(img, invalid_meta, decode=False)

    invalid_future_meta = _build_dummy_img_metas(
        batch_size=batch_size,
        seq_len=seq_len,
        include_time_indices=True,
        include_future_time_stamps=True,
        pred_horizon=cfg.pred_horizon,
        future_dt=cfg.future_dt,
    )
    invalid_future_meta[0]["future_time_stamps"][2] = invalid_future_meta[0]["future_time_stamps"][1] - 0.1
    with pytest.raises(ValueError, match="strictly increasing"):
        model(img, invalid_future_meta, decode=False)


def test_decode_parity_mode_score_and_topk_semantics(instrumented_debug_forward):
    cfg = instrumented_debug_forward["cfg"]
    outputs = instrumented_debug_forward["outputs"]
    decoded_payload = instrumented_debug_forward["decoded"]["decoded"]
    batch_size = outputs["mode_probs"].shape[0]
    topk = min(cfg.decode_topk, cfg.num_modes)

    expected_keys = {
        "best_mode_idx",
        "best_mode_prob",
        "best_trajectory",
        "topk_mode_idx",
        "topk_mode_prob",
        "topk_trajectory",
    }
    assert set(decoded_payload.keys()) == expected_keys
    assert decoded_payload["topk_mode_idx"].shape == (batch_size, topk)
    assert decoded_payload["topk_mode_prob"].shape == (batch_size, topk)
    assert decoded_payload["topk_trajectory"].shape == (batch_size, topk, cfg.pred_horizon, 2)

    expected_topk_prob, expected_topk_idx = torch.topk(outputs["mode_probs"], k=topk, dim=-1)
    assert torch.equal(decoded_payload["topk_mode_idx"], expected_topk_idx)
    assert torch.allclose(decoded_payload["topk_mode_prob"], expected_topk_prob, atol=1e-6, rtol=1e-6)

    gather_topk = expected_topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, cfg.pred_horizon, 2)
    expected_topk_trajectory = outputs["trajectory_preds"].gather(dim=1, index=gather_topk)
    assert torch.allclose(
        decoded_payload["topk_trajectory"],
        expected_topk_trajectory,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.equal(decoded_payload["best_mode_idx"], expected_topk_idx[:, 0])
    assert torch.allclose(decoded_payload["best_mode_prob"], expected_topk_prob[:, 0], atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        decoded_payload["best_trajectory"],
        expected_topk_trajectory[:, 0],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.equal(decoded_payload["best_mode_idx"], outputs["mode_probs"].argmax(dim=-1))


def test_temporal_validity_mask_parity_and_update_behavior():
    cfg = debug_forward_config(pred_horizon=6)
    model = BEVerseLite(cfg).eval()

    batch_size = 1
    seq_len = 3
    img = torch.zeros(batch_size, seq_len, cfg.num_cams, 3, 64, 96)
    img[:, 0] = -3.0
    img[:, 1] = 5.0
    img[:, 2] = 0.25

    base_metas = _build_dummy_img_metas(
        batch_size=batch_size,
        seq_len=seq_len,
        include_time_indices=True,
    )
    img_is_valid = torch.tensor([[True, False, True]], dtype=torch.bool)
    metas_with_validity = [dict(base_metas[0], img_is_valid=img_is_valid[0].tolist())]

    with torch.no_grad():
        outputs_from_meta = model(img, metas_with_validity, decode=False)
        outputs_from_tensor = model(img, base_metas, img_is_valid=img_is_valid, decode=False)
        outputs_all_valid = model(img, base_metas, img_is_valid=torch.ones_like(img_is_valid), decode=False)

    assert torch.allclose(
        outputs_from_meta["temporal_tokens"],
        outputs_from_tensor["temporal_tokens"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        outputs_from_meta["trajectory_preds"],
        outputs_from_tensor["trajectory_preds"],
        atol=1e-6,
        rtol=1e-6,
    )

    temporal_delta = (outputs_from_tensor["temporal_tokens"] - outputs_all_valid["temporal_tokens"]).abs().max()
    assert float(temporal_delta) > 1e-6


def test_trajectory_consistency_matches_cumsum_deltas(instrumented_debug_forward):
    outputs = instrumented_debug_forward["outputs"]
    reconstructed = outputs["trajectory_deltas"].cumsum(dim=2)
    assert torch.allclose(outputs["trajectory_preds"], reconstructed, atol=1e-5, rtol=1e-5)


def test_prediction_metric_smoke(instrumented_debug_forward):
    outputs = instrumented_debug_forward["outputs"]
    batch_size = outputs["trajectory_preds"].shape[0]
    horizon = outputs["trajectory_preds"].shape[2]

    target = outputs["trajectory_preds"][:, 0] + 0.1
    valid_mask = torch.ones((batch_size, horizon), dtype=target.dtype, device=target.device)
    valid_mask[:, -1] = 0.0

    best_mode_idx, best_traj, best_ade = select_best_mode_by_ade(
        outputs["trajectory_preds"], target, valid_mask=valid_mask
    )
    ade, fde = compute_ade_fde(best_traj, target, valid_mask=valid_mask)

    assert best_mode_idx.shape == (batch_size,)
    assert best_traj.shape == (batch_size, horizon, 2)
    assert best_ade.shape == (batch_size,)
    assert ade.shape == (batch_size,)
    assert fde.shape == (batch_size,)
    assert torch.isfinite(best_ade).all()
    assert torch.isfinite(ade).all()
    assert torch.isfinite(fde).all()
    assert (ade >= 0).all()
    assert (fde >= 0).all()
