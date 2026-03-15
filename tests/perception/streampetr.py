"""Intermediate tensor validation tests for pure-PyTorch StreamPETR."""

from __future__ import annotations

import copy
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from pytorch_implementation.perception.streampetr.config import debug_forward_config
from pytorch_implementation.perception.streampetr.model import StreamPETRLite
from tests._shared.hook_helpers import register_hook_append
from tests._shared.parity_helpers import assert_decoded_topk_label_score_consistency
from tests._shared.tensor_helpers import conv2d_out, first_tensor, iter_tensors


def _build_dummy_img_metas(
    batch_size: int,
    num_cams: int,
    height: int,
    width: int,
) -> list[dict[str, Any]]:
    metas = []
    for batch_idx in range(batch_size):
        lidar2img = []
        for cam_idx in range(num_cams):
            projection = [
                [1.0, 0.0, 0.0, float(width) * (0.5 + cam_idx * 0.01)],
                [0.0, 1.0, 0.0, float(height) * 0.5],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            lidar2img.append(projection)
        metas.append(
            {
                "sample_idx": f"sample_{batch_idx}",
                "scene_token": f"scene_{batch_idx}",
                "lidar2img": lidar2img,
                "img_shape": [(height, width, 3) for _ in range(num_cams)],
                "pad_shape": [(height, width, 3) for _ in range(num_cams)],
            }
        )
    return metas


def _first_tensor(value: Any) -> torch.Tensor | None:
    return first_tensor(value)


def _iter_tensors(value: Any):
    yield from iter_tensors(value)


def _register_hook(module, name: str, capture: dict[str, list[Any]], handles: list) -> None:
    register_hook_append(module, name, capture, handles)


def _assert_decoded_topk_label_score_consistency(
    cls_scores: torch.Tensor,
    decoded_scores: torch.Tensor,
    decoded_labels: torch.Tensor,
    *,
    max_num: int,
    num_classes: int,
) -> None:
    assert_decoded_topk_label_score_consistency(
        cls_scores,
        decoded_scores,
        decoded_labels,
        max_num=max_num,
        num_classes=num_classes,
    )


def _conv2d_out(size: int, kernel: int, stride: int, padding: int) -> int:
    return conv2d_out(size, kernel, stride, padding)


@pytest.fixture()
def instrumented_debug_forward():
    cfg = debug_forward_config(
        num_queries=48,
        decoder_layers=2,
        depth_num=6,
        memory_len=40,
        topk_proposals=12,
        num_propagated=8,
    )
    model = StreamPETRLite(cfg).eval()

    batch_size = 1
    height, width = 96, 160
    img1 = torch.randn(batch_size, cfg.num_cams, 3, height, width)
    img2 = torch.randn(batch_size, cfg.num_cams, 3, height, width)
    img_metas = _build_dummy_img_metas(batch_size=batch_size, num_cams=cfg.num_cams, height=height, width=width)

    capture: dict[str, list[Any]] = {}
    handles = []

    # Backbone/FPN hooks.
    _register_hook(model.backbone_neck.backbone.stem, "backbone.stem", capture, handles)
    for idx, stage in enumerate(model.backbone_neck.backbone.stages):
        _register_hook(stage, f"backbone.stage{idx}", capture, handles)
    _register_hook(model.backbone_neck.neck.output_convs[0], "fpn.output0", capture, handles)

    # Head hooks.
    _register_hook(model.head.input_proj, "head.input_proj", capture, handles)
    _register_hook(model.head.position_encoder, "head.position_encoder", capture, handles)
    _register_hook(model.head.adapt_pos3d, "head.adapt_pos3d", capture, handles)
    _register_hook(model.head.reference_points, "head.reference_points", capture, handles)
    _register_hook(model.head.query_embedding, "head.query_embedding", capture, handles)
    _register_hook(model.head.time_embedding, "head.time_embedding", capture, handles)

    for idx, layer in enumerate(model.head.transformer.decoder.layers):
        _register_hook(layer, f"decoder.layer{idx}", capture, handles)
        _register_hook(layer.self_attn, f"decoder.layer{idx}.self_attn", capture, handles)
        _register_hook(layer.cross_attn, f"decoder.layer{idx}.cross_attn", capture, handles)
        _register_hook(layer.ffn, f"decoder.layer{idx}.ffn", capture, handles)

    for idx, branch in enumerate(model.head.cls_branches):
        _register_hook(branch, f"head.cls_branch{idx}", capture, handles)
    for idx, branch in enumerate(model.head.reg_branches):
        _register_hook(branch, f"head.reg_branch{idx}", capture, handles)

    with torch.no_grad():
        outputs_first = model(img1, img_metas, decode=False, prev_exists=torch.zeros(batch_size))
        memory_after_first = {
            "embedding": model.head.memory_embedding.clone(),
            "reference": model.head.memory_reference_point.clone(),
            "timestamp": model.head.memory_timestamp.clone(),
        }
        outputs_second = model(img2, img_metas, decode=False, prev_exists=torch.ones(batch_size))
        memory_after_second = {
            "embedding": model.head.memory_embedding.clone(),
            "reference": model.head.memory_reference_point.clone(),
            "timestamp": model.head.memory_timestamp.clone(),
        }
        outputs_reset = model(img2, img_metas, decode=False, prev_exists=torch.zeros(batch_size))
        memory_after_reset = {
            "embedding": model.head.memory_embedding.clone(),
            "reference": model.head.memory_reference_point.clone(),
            "timestamp": model.head.memory_timestamp.clone(),
        }

    for handle in handles:
        handle.remove()

    assert isinstance(outputs_first, dict)
    assert isinstance(outputs_second, dict)
    return {
        "cfg": cfg,
        "model": model,
        "img1": img1,
        "img2": img2,
        "img_metas": img_metas,
        "batch_size": batch_size,
        "height": height,
        "width": width,
        "capture": capture,
        "outputs_first": outputs_first,
        "outputs_second": outputs_second,
        "outputs_reset": outputs_reset,
        "memory_after_first": memory_after_first,
        "memory_after_second": memory_after_second,
        "memory_after_reset": memory_after_reset,
    }


def test_intermediate_hooks_cover_all_major_layers(instrumented_debug_forward):
    data = instrumented_debug_forward
    cfg = data["cfg"]
    capture = data["capture"]

    expected_names = {
        "backbone.stem",
        "backbone.stage0",
        "backbone.stage1",
        "backbone.stage2",
        "backbone.stage3",
        "fpn.output0",
        "head.input_proj",
        "head.position_encoder",
        "head.adapt_pos3d",
        "head.reference_points",
        "head.query_embedding",
        "head.time_embedding",
    }
    for idx in range(cfg.num_decoder_layers):
        expected_names.update(
            {
                f"decoder.layer{idx}",
                f"decoder.layer{idx}.self_attn",
                f"decoder.layer{idx}.cross_attn",
                f"decoder.layer{idx}.ffn",
                f"head.cls_branch{idx}",
                f"head.reg_branch{idx}",
            }
        )

    missing = sorted(expected_names - set(capture.keys()))
    assert not missing, f"Missing intermediate captures: {missing}"
    for name in expected_names:
        assert capture[name], f"Hook for '{name}' captured no values."


def test_intermediate_shapes_match_debug_config(instrumented_debug_forward):
    data = instrumented_debug_forward
    cfg = data["cfg"]
    capture = data["capture"]
    outputs = data["outputs_second"]
    memory_after_first = data["memory_after_first"]
    memory_after_second = data["memory_after_second"]
    memory_after_reset = data["memory_after_reset"]
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

    assert _first_tensor(capture["backbone.stem"][-1]).shape == (batch_size * cams, c1, stem_h, stem_w)
    assert _first_tensor(capture["backbone.stage0"][-1]).shape == (batch_size * cams, c1, stem_h, stem_w)
    assert _first_tensor(capture["backbone.stage1"][-1]).shape == (batch_size * cams, c2, stage1_h, stage1_w)
    assert _first_tensor(capture["backbone.stage2"][-1]).shape == (batch_size * cams, c3, stage2_h, stage2_w)
    assert _first_tensor(capture["backbone.stage3"][-1]).shape == (batch_size * cams, c4, stage3_h, stage3_w)
    assert _first_tensor(capture["fpn.output0"][-1]).shape == (
        batch_size * cams,
        cfg.backbone_neck.out_channels,
        stage3_h,
        stage3_w,
    )

    expected_q = cfg.num_queries
    expected_c = cfg.embed_dims
    expected_l = cfg.num_decoder_layers
    expected_q_total = expected_q + min(cfg.num_propagated, cfg.memory_len)

    assert _first_tensor(capture["head.input_proj"][-1]).shape == (
        batch_size * cams,
        expected_c,
        stage3_h,
        stage3_w,
    )
    assert _first_tensor(capture["head.position_encoder"][-1]).shape == (
        batch_size * cams,
        expected_c,
        stage3_h,
        stage3_w,
    )
    assert _first_tensor(capture["head.adapt_pos3d"][-1]).shape == (
        batch_size * cams,
        expected_c,
        stage3_h,
        stage3_w,
    )
    assert _first_tensor(capture["head.reference_points"][-1]).shape == (expected_q, 3)

    query_embed_shape = _first_tensor(capture["head.query_embedding"][-1]).shape
    time_embed_shape = _first_tensor(capture["head.time_embedding"][-1]).shape
    assert query_embed_shape[-1] == expected_c
    assert time_embed_shape[-1] == expected_c

    for idx in range(expected_l):
        layer_out = _first_tensor(capture[f"decoder.layer{idx}"][-1])
        self_attn_out = _first_tensor(capture[f"decoder.layer{idx}.self_attn"][-1])
        cross_attn_out = _first_tensor(capture[f"decoder.layer{idx}.cross_attn"][-1])
        ffn_out = _first_tensor(capture[f"decoder.layer{idx}.ffn"][-1])
        assert layer_out.shape == (expected_q_total, batch_size, expected_c)
        assert self_attn_out.shape == (expected_q_total, batch_size, expected_c)
        assert cross_attn_out.shape == (expected_q_total, batch_size, expected_c)
        assert ffn_out.shape == (expected_q_total, batch_size, expected_c)
        assert _first_tensor(capture[f"head.cls_branch{idx}"][-1]).shape == (batch_size, expected_q, cfg.num_classes)
        assert _first_tensor(capture[f"head.reg_branch{idx}"][-1]).shape == (batch_size, expected_q, cfg.code_size)

    assert outputs["all_cls_scores"].shape == (expected_l, batch_size, expected_q, cfg.num_classes)
    assert outputs["all_bbox_preds"].shape == (expected_l, batch_size, expected_q, cfg.code_size)

    # Memory bank state is persisted with fixed-length tensors and updated between frames.
    assert memory_after_first["embedding"].shape == (batch_size, cfg.memory_len, expected_c)
    assert memory_after_first["reference"].shape == (batch_size, cfg.memory_len, 3)
    assert memory_after_first["timestamp"].shape == (batch_size, cfg.memory_len, 1)
    assert memory_after_second["embedding"].shape == (batch_size, cfg.memory_len, expected_c)
    assert not torch.allclose(memory_after_first["embedding"], memory_after_second["embedding"])
    assert memory_after_second["reference"].min() >= 0.0
    assert memory_after_second["reference"].max() <= 1.0
    topk = min(cfg.topk_proposals, cfg.memory_len, cfg.num_queries)
    assert torch.allclose(
        memory_after_second["timestamp"][:, :topk],
        torch.zeros_like(memory_after_second["timestamp"][:, :topk]),
    )
    if cfg.memory_len > topk:
        assert torch.all(memory_after_second["timestamp"][:, topk:] >= 1.0)
    assert torch.allclose(memory_after_reset["timestamp"], torch.zeros_like(memory_after_reset["timestamp"]))
    assert memory_after_reset["reference"].min() >= 0.0
    assert memory_after_reset["reference"].max() <= 1.0
    assert not torch.allclose(memory_after_second["embedding"], memory_after_reset["embedding"])


def test_metadata_contract_validation_requires_scene_and_geometry_keys():
    cfg = debug_forward_config(
        num_queries=32,
        decoder_layers=2,
        depth_num=6,
        memory_len=24,
        topk_proposals=8,
        num_propagated=6,
    )
    model = StreamPETRLite(cfg).eval()
    batch_size = 1
    height, width = 96, 160
    img = torch.randn(batch_size, cfg.num_cams, 3, height, width)
    img_metas = _build_dummy_img_metas(batch_size=batch_size, num_cams=cfg.num_cams, height=height, width=width)

    with torch.no_grad():
        outputs = model(img, img_metas, decode=False, prev_exists=torch.zeros(batch_size))
    assert isinstance(outputs, dict)

    missing_scene_token = copy.deepcopy(img_metas)
    missing_scene_token[0].pop("scene_token")
    with pytest.raises(KeyError, match="scene_token"):
        model(img, missing_scene_token, decode=False)

    with torch.no_grad():
        explicit_prev_outputs = model(img, missing_scene_token, decode=False, prev_exists=torch.zeros(batch_size))
    assert isinstance(explicit_prev_outputs, dict)

    bad_lidar2img = copy.deepcopy(img_metas)
    bad_lidar2img[0]["lidar2img"] = bad_lidar2img[0]["lidar2img"][:-1]
    with pytest.raises(ValueError, match="lidar2img"):
        model(img, bad_lidar2img, decode=False, prev_exists=torch.zeros(batch_size))


def test_decode_contract_semantics_and_topk_consistency(instrumented_debug_forward):
    data = instrumented_debug_forward
    cfg = data["cfg"]
    model = data["model"]

    with torch.no_grad():
        decoded_pack = model(data["img2"], data["img_metas"], decode=True, prev_exists=torch.ones(data["batch_size"]))

    assert isinstance(decoded_pack, dict)
    assert set(decoded_pack.keys()) == {"preds", "decoded"}
    preds = decoded_pack["preds"]
    decoded = decoded_pack["decoded"]
    assert isinstance(preds, dict)
    assert isinstance(decoded, list)
    assert len(decoded) == data["batch_size"]

    cls_last = preds["all_cls_scores"][-1]
    for batch_idx, sample in enumerate(decoded):
        assert set(sample.keys()) == {"bboxes", "scores", "labels"}
        assert sample["bboxes"].ndim == 2
        assert sample["bboxes"].shape[1] == 9
        assert sample["scores"].ndim == 1
        assert sample["labels"].ndim == 1
        assert sample["scores"].shape == sample["labels"].shape
        assert sample["scores"].shape[0] == sample["bboxes"].shape[0]
        assert sample["labels"].dtype == torch.long
        _assert_decoded_topk_label_score_consistency(
            cls_last[batch_idx],
            sample["scores"],
            sample["labels"],
            max_num=cfg.max_num,
            num_classes=cfg.num_classes,
        )


def test_intermediate_and_final_tensors_are_finite(instrumented_debug_forward):
    data = instrumented_debug_forward
    capture = data["capture"]
    outputs_first = data["outputs_first"]
    outputs_second = data["outputs_second"]
    outputs_reset = data["outputs_reset"]
    memory_after_first = data["memory_after_first"]
    memory_after_second = data["memory_after_second"]
    memory_after_reset = data["memory_after_reset"]

    for name, values in capture.items():
        assert values, f"No values captured for '{name}'."
        for value in values:
            tensors = list(_iter_tensors(value))
            assert tensors, f"No tensor found in captured output for '{name}'."
            for tensor in tensors:
                assert torch.isfinite(tensor).all(), f"Non-finite values found in intermediate '{name}'."

    for output_name, output_dict in (("first", outputs_first), ("second", outputs_second), ("reset", outputs_reset)):
        for name, value in output_dict.items():
            if value is None:
                continue
            tensors = list(_iter_tensors(value))
            assert tensors, f"No tensor found in output '{output_name}:{name}'."
            for tensor in tensors:
                assert torch.isfinite(tensor).all(), f"Non-finite values found in final output '{output_name}:{name}'."

    for bank_name, bank in (("first", memory_after_first), ("second", memory_after_second), ("reset", memory_after_reset)):
        for name, value in bank.items():
            tensors = list(_iter_tensors(value))
            assert tensors, f"No tensor found in memory bank '{bank_name}:{name}'."
            for tensor in tensors:
                assert torch.isfinite(tensor).all(), f"Non-finite values found in memory bank '{bank_name}:{name}'."

