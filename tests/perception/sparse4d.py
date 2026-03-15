"""Intermediate tensor validation tests for pure-PyTorch Sparse4D."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from pytorch_implementation.perception.sparse4d.config import debug_forward_config
from pytorch_implementation.perception.sparse4d.model import Sparse4DLite


def _build_dummy_metas(
    batch_size: int,
    num_cams: int,
    height: int,
    width: int,
    *,
    timestamp: float | None = None,
) -> dict[str, torch.Tensor]:
    projection_mat = torch.eye(4, dtype=torch.float32).view(1, 1, 4, 4).repeat(batch_size, num_cams, 1, 1)
    for cam_idx in range(num_cams):
        projection_mat[:, cam_idx, 0, 3] = width * (0.45 + 0.02 * cam_idx)
        projection_mat[:, cam_idx, 1, 3] = height * 0.5
    image_wh = torch.tensor([width, height], dtype=torch.float32).view(1, 1, 2).repeat(batch_size, num_cams, 1)
    metas: dict[str, torch.Tensor] = {"projection_mat": projection_mat, "image_wh": image_wh}
    if timestamp is not None:
        metas["timestamp"] = torch.full((batch_size,), float(timestamp), dtype=torch.float32)
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


def _assert_decoded_topk_label_score_consistency(
    cls_scores: torch.Tensor,
    decoded_scores: torch.Tensor,
    decoded_labels: torch.Tensor,
    *,
    max_num: int,
    num_classes: int,
) -> None:
    flat_scores = cls_scores.sigmoid().reshape(-1)
    topk = min(int(max_num), int(flat_scores.numel()))
    topk_scores, topk_indices = flat_scores.topk(topk)
    topk_labels = (topk_indices % num_classes).to(dtype=torch.long)

    assert decoded_scores.ndim == 1
    assert decoded_labels.ndim == 1
    assert decoded_scores.shape[0] == decoded_labels.shape[0]
    assert decoded_scores.shape[0] <= topk
    if decoded_scores.numel() > 1:
        assert torch.all(decoded_scores[:-1] >= decoded_scores[1:])
    assert torch.all((decoded_scores >= 0.0) & (decoded_scores <= 1.0))
    if decoded_labels.numel() > 0:
        assert decoded_labels.dtype == torch.long
        assert int(decoded_labels.min().item()) >= 0
        assert int(decoded_labels.max().item()) < num_classes

    remaining = [(float(score), int(label)) for score, label in zip(topk_scores.tolist(), topk_labels.tolist())]
    for score, label in zip(decoded_scores.tolist(), decoded_labels.tolist()):
        matched_idx = None
        for idx, (candidate_score, candidate_label) in enumerate(remaining):
            if candidate_label == int(label) and abs(candidate_score - float(score)) <= 1e-6:
                matched_idx = idx
                break
        assert matched_idx is not None, "Decoded score/label pair is inconsistent with top-k logits."
        remaining.pop(matched_idx)


def _conv2d_out(size: int, kernel: int, stride: int, padding: int) -> int:
    return ((size + 2 * padding - kernel) // stride) + 1


@pytest.fixture()
def instrumented_debug_forward():
    cfg = replace(
        debug_forward_config(num_queries=48, decoder_layers=2),
        num_temp_instances=12,
        num_single_frame_decoder=1,
    )
    model = Sparse4DLite(cfg).eval()

    batch_size = 1
    height, width = 96, 160
    img = torch.randn(batch_size, cfg.num_cams, 3, height, width)
    img_second = torch.randn(batch_size, cfg.num_cams, 3, height, width)
    metas = _build_dummy_metas(
        batch_size=batch_size,
        num_cams=cfg.num_cams,
        height=height,
        width=width,
        timestamp=0.0,
    )
    metas_second = _build_dummy_metas(
        batch_size=batch_size,
        num_cams=cfg.num_cams,
        height=height,
        width=width,
        timestamp=1.0,
    )

    capture: dict[str, Any] = {}
    handles = []

    _register_hook(model.backbone_neck.backbone.stem, "backbone.stem", capture, handles)
    for idx, stage in enumerate(model.backbone_neck.backbone.stages):
        _register_hook(stage, f"backbone.stage{idx}", capture, handles)
    for idx, conv in enumerate(model.backbone_neck.neck.output_convs):
        _register_hook(conv, f"neck.output{idx}", capture, handles)

    _register_hook(model.head.instance_bank, "head.instance_bank", capture, handles)
    _register_hook(model.head.anchor_encoder, "head.anchor_encoder", capture, handles)

    for idx, layer in enumerate(model.head.decoder.layers):
        _register_hook(layer, f"decoder.layer{idx}", capture, handles)
        _register_hook(layer.self_attn, f"decoder.layer{idx}.self_attn", capture, handles)
        _register_hook(layer.cross_attn, f"decoder.layer{idx}.cross_attn", capture, handles)
        _register_hook(layer.ffn, f"decoder.layer{idx}.ffn", capture, handles)
    for idx, refine in enumerate(model.head.decoder.refinement_layers):
        _register_hook(refine.cls_branch, f"decoder.refine{idx}.cls_branch", capture, handles)
        _register_hook(refine.reg_branch, f"decoder.refine{idx}.reg_branch", capture, handles)

    with torch.no_grad():
        outputs = model(img, metas, decode=False)
        cache_after_first = {
            "feature": model.head.instance_bank.cached_feature.clone(),
            "anchor": model.head.instance_bank.cached_anchor.clone(),
            "confidence": model.head.instance_bank.confidence.clone(),
        }
        outputs_second = model(img_second, metas_second, decode=False)
        cache_after_second = {
            "feature": model.head.instance_bank.cached_feature.clone(),
            "anchor": model.head.instance_bank.cached_anchor.clone(),
            "confidence": model.head.instance_bank.confidence.clone(),
        }
        instance_bank_mask = None
        if model.head.instance_bank.mask is not None:
            instance_bank_mask = model.head.instance_bank.mask.clone()

    for handle in handles:
        handle.remove()

    assert isinstance(outputs, dict)
    assert isinstance(outputs_second, dict)
    return {
        "cfg": cfg,
        "model": model,
        "img": img,
        "img_second": img_second,
        "metas": metas,
        "metas_second": metas_second,
        "batch_size": batch_size,
        "height": height,
        "width": width,
        "capture": capture,
        "outputs": outputs,
        "outputs_second": outputs_second,
        "cache_after_first": cache_after_first,
        "cache_after_second": cache_after_second,
        "instance_bank_mask": instance_bank_mask,
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
        "neck.output0",
        "neck.output1",
        "neck.output2",
        "neck.output3",
        "head.instance_bank",
        "head.anchor_encoder",
    }
    for idx in range(cfg.num_decoder_layers):
        expected_names.update(
            {
                f"decoder.layer{idx}",
                f"decoder.layer{idx}.self_attn",
                f"decoder.layer{idx}.cross_attn",
                f"decoder.layer{idx}.ffn",
                f"decoder.refine{idx}.cls_branch",
                f"decoder.refine{idx}.reg_branch",
            }
        )

    missing = sorted(expected_names - set(capture.keys()))
    assert not missing, f"Missing intermediate captures: {missing}"


def test_intermediate_shapes_match_debug_config(instrumented_debug_forward):
    data = instrumented_debug_forward
    cfg = data["cfg"]
    model = data["model"]
    img = data["img"]
    metas = data["metas"]
    capture = data["capture"]
    outputs = data["outputs"]
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

    expected_c = cfg.embed_dims
    assert _first_tensor(capture["neck.output0"]).shape == (batch_size * cams, expected_c, stem_h, stem_w)
    assert _first_tensor(capture["neck.output1"]).shape == (batch_size * cams, expected_c, stage1_h, stage1_w)
    assert _first_tensor(capture["neck.output2"]).shape == (batch_size * cams, expected_c, stage2_h, stage2_w)
    assert _first_tensor(capture["neck.output3"]).shape == (batch_size * cams, expected_c, stage3_h, stage3_w)

    expected_q = cfg.num_queries
    expected_l = cfg.num_decoder_layers
    assert _first_tensor(capture["head.instance_bank"]).shape == (batch_size, expected_q, expected_c)
    assert _first_tensor(capture["head.anchor_encoder"]).shape == (batch_size, expected_q, expected_c)

    for idx in range(expected_l):
        assert _first_tensor(capture[f"decoder.layer{idx}"]).shape == (batch_size, expected_q, expected_c)
        assert _first_tensor(capture[f"decoder.layer{idx}.self_attn"]).shape == (
            batch_size,
            expected_q,
            expected_c,
        )
        assert _first_tensor(capture[f"decoder.layer{idx}.cross_attn"]).shape == (
            batch_size,
            expected_q,
            expected_c,
        )
        assert _first_tensor(capture[f"decoder.layer{idx}.ffn"]).shape == (batch_size, expected_q, expected_c)
        assert _first_tensor(capture[f"decoder.refine{idx}.cls_branch"]).shape == (
            batch_size,
            expected_q,
            cfg.num_classes,
        )
        assert _first_tensor(capture[f"decoder.refine{idx}.reg_branch"]).shape == (
            batch_size,
            expected_q,
            cfg.box_code_size,
        )

    assert outputs["all_cls_scores"].shape == (expected_l, batch_size, expected_q, cfg.num_classes)
    assert outputs["all_bbox_preds"].shape == (expected_l, batch_size, expected_q, cfg.box_code_size)

    with torch.no_grad():
        decoded = model(img, metas, decode=True)["decoded"]
    assert len(decoded) == batch_size
    assert decoded[0]["boxes_3d"].shape[1] == 10
    assert decoded[0]["scores_3d"].shape[0] <= cfg.max_detections
    assert decoded[0]["labels_3d"].dtype == torch.long


def test_metadata_contract_validation_checks_optional_tensor_shapes():
    cfg = replace(debug_forward_config(num_queries=24, decoder_layers=2), num_temp_instances=4)
    model = Sparse4DLite(cfg).eval()
    batch_size = 1
    height, width = 96, 160
    img = torch.randn(batch_size, cfg.num_cams, 3, height, width)
    metas = _build_dummy_metas(batch_size=batch_size, num_cams=cfg.num_cams, height=height, width=width)

    with torch.no_grad():
        outputs = model(img, metas, decode=False)
    assert isinstance(outputs, dict)

    bad_projection = {
        "projection_mat": torch.eye(4, dtype=torch.float32).view(1, 4, 4),
        "image_wh": metas["image_wh"],
    }
    with pytest.raises(ValueError, match="projection_mat"):
        model(img, bad_projection, decode=False)

    bad_image_wh = {
        "projection_mat": metas["projection_mat"],
        "image_wh": torch.ones(batch_size, cfg.num_cams, dtype=torch.float32),
    }
    with pytest.raises(ValueError, match="image_wh"):
        model(img, bad_image_wh, decode=False)


def test_decode_contract_semantics_and_topk_consistency(instrumented_debug_forward):
    data = instrumented_debug_forward
    cfg = data["cfg"]
    model = data["model"]

    with torch.no_grad():
        decoded_pack = model(data["img_second"], data["metas_second"], decode=True)

    assert isinstance(decoded_pack, dict)
    assert set(decoded_pack.keys()) == {"preds", "decoded"}
    preds = decoded_pack["preds"]
    decoded = decoded_pack["decoded"]
    assert isinstance(preds, dict)
    assert isinstance(decoded, list)
    assert len(decoded) == data["batch_size"]

    cls_last = preds["all_cls_scores"][-1]
    for batch_idx, sample in enumerate(decoded):
        assert set(sample.keys()) == {"boxes_3d", "scores_3d", "labels_3d"}
        assert sample["boxes_3d"].ndim == 2
        assert sample["boxes_3d"].shape[1] == 10
        assert sample["scores_3d"].ndim == 1
        assert sample["labels_3d"].ndim == 1
        assert sample["scores_3d"].shape == sample["labels_3d"].shape
        assert sample["scores_3d"].shape[0] == sample["boxes_3d"].shape[0]
        assert sample["labels_3d"].dtype == torch.long
        _assert_decoded_topk_label_score_consistency(
            cls_last[batch_idx],
            sample["scores_3d"],
            sample["labels_3d"],
            max_num=cfg.max_detections,
            num_classes=cfg.num_classes,
        )


def test_instance_bank_cache_and_update_contract(instrumented_debug_forward):
    data = instrumented_debug_forward
    cfg = data["cfg"]
    batch_size = data["batch_size"]
    cache_first = data["cache_after_first"]
    cache_second = data["cache_after_second"]
    mask = data["instance_bank_mask"]

    assert cfg.num_temp_instances > 0
    assert cache_first["feature"].shape == (batch_size, cfg.num_temp_instances, cfg.embed_dims)
    assert cache_first["anchor"].shape == (batch_size, cfg.num_temp_instances, cfg.box_code_size)
    assert cache_first["confidence"].shape == (batch_size, cfg.num_temp_instances)
    assert cache_second["feature"].shape == cache_first["feature"].shape
    assert cache_second["anchor"].shape == cache_first["anchor"].shape
    assert cache_second["confidence"].shape == cache_first["confidence"].shape
    assert torch.all(cache_first["confidence"][:, :-1] >= cache_first["confidence"][:, 1:])
    assert torch.all(cache_second["confidence"][:, :-1] >= cache_second["confidence"][:, 1:])
    assert mask is not None
    assert mask.shape == (batch_size,)
    assert bool(mask.all())
    assert not torch.allclose(cache_first["feature"], cache_second["feature"])
    assert not torch.allclose(cache_first["anchor"], cache_second["anchor"])
    assert not torch.allclose(data["outputs"]["all_bbox_preds"][-1], data["outputs_second"]["all_bbox_preds"][-1])


def test_intermediate_and_final_tensors_are_finite(instrumented_debug_forward):
    data = instrumented_debug_forward
    capture = data["capture"]
    outputs = data["outputs"]

    for name, value in capture.items():
        tensors = list(_iter_tensors(value))
        assert tensors, f"No tensor found in captured output for '{name}'."
        for tensor in tensors:
            assert torch.isfinite(tensor).all(), f"Non-finite values found in intermediate '{name}'."

    for name, value in outputs.items():
        tensors = list(_iter_tensors(value))
        assert tensors, f"No tensor found in output '{name}'."
        for tensor in tensors:
            assert torch.isfinite(tensor).all(), f"Non-finite values found in final output '{name}'."
