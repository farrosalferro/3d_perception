"""Intermediate tensor validation tests for pure-PyTorch MapTR."""

from __future__ import annotations

import copy
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from pytorch_implementation.perception.maptr.config import debug_forward_config
from pytorch_implementation.perception.maptr.model import MapTRLite


def _build_dummy_img_metas(
    batch_size: int,
    num_cams: int,
    height: int,
    width: int,
) -> list[dict[str, Any]]:
    metas = []
    for batch_idx in range(batch_size):
        metas.append(
            {
                "sample_idx": f"sample_{batch_idx}",
                "img_shape": [(height, width, 3) for _ in range(num_cams)],
                "pad_shape": [(height, width, 3) for _ in range(num_cams)],
            }
        )
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
    cfg = debug_forward_config(num_vec=10, num_pts_per_vec=4, decoder_layers=2)
    model = MapTRLite(cfg).eval()

    batch_size = 1
    height, width = 96, 160
    img = torch.randn(batch_size, cfg.num_cams, 3, height, width)
    img_metas = _build_dummy_img_metas(batch_size=batch_size, num_cams=cfg.num_cams, height=height, width=width)

    capture: dict[str, Any] = {}
    handles = []

    # Backbone/FPN hooks.
    _register_hook(model.backbone_neck.backbone.stem, "backbone.stem", capture, handles)
    for idx, stage in enumerate(model.backbone_neck.backbone.stages):
        _register_hook(stage, f"backbone.stage{idx}", capture, handles)
    _register_hook(model.backbone_neck.neck.output_convs[0], "fpn.output0", capture, handles)

    # Head + transformer hooks.
    _register_hook(model.head.input_proj, "head.input_proj", capture, handles)
    _register_hook(model.head.adapt_pos3d, "head.adapt_pos3d", capture, handles)
    _register_hook(model.head.bev_embedding, "head.bev_embedding", capture, handles)
    _register_hook(model.head.instance_embedding, "head.instance_embedding", capture, handles)
    _register_hook(model.head.pts_embedding, "head.pts_embedding", capture, handles)
    _register_hook(model.head.transformer.bev_encoder.cross_attn, "bev_encoder.cross_attn", capture, handles)
    _register_hook(model.head.transformer.bev_encoder.ffn, "bev_encoder.ffn", capture, handles)

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
        outputs = model(img, img_metas, decode=False)

    for handle in handles:
        handle.remove()

    assert isinstance(outputs, dict)
    return {
        "cfg": cfg,
        "model": model,
        "img": img,
        "img_metas": img_metas,
        "batch_size": batch_size,
        "height": height,
        "width": width,
        "capture": capture,
        "outputs": outputs,
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
        "head.adapt_pos3d",
        "head.bev_embedding",
        "head.instance_embedding",
        "head.pts_embedding",
        "bev_encoder.cross_attn",
        "bev_encoder.ffn",
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


def test_intermediate_shapes_match_debug_config(instrumented_debug_forward):
    data = instrumented_debug_forward
    cfg = data["cfg"]
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
    assert _first_tensor(capture["fpn.output0"]).shape == (
        batch_size * cams,
        cfg.backbone_neck.out_channels,
        stage3_h,
        stage3_w,
    )

    expected_q = cfg.num_query
    expected_c = cfg.embed_dims
    expected_l = cfg.num_decoder_layers
    expected_v = cfg.num_vec
    expected_p = cfg.num_pts_per_vec
    expected_bev = cfg.bev_h * cfg.bev_w

    assert _first_tensor(capture["head.input_proj"]).shape == (
        batch_size * cams,
        expected_c,
        stage3_h,
        stage3_w,
    )
    assert _first_tensor(capture["head.adapt_pos3d"]).shape == (
        batch_size * cams,
        expected_c,
        stage3_h,
        stage3_w,
    )
    assert _first_tensor(capture["head.bev_embedding"]).shape == (expected_bev, expected_c)
    assert _first_tensor(capture["head.instance_embedding"]).shape == (expected_v, expected_c)
    assert _first_tensor(capture["head.pts_embedding"]).shape == (expected_p, expected_c)

    bev_cross_attn_out = _first_tensor(capture["bev_encoder.cross_attn"])
    bev_ffn_out = _first_tensor(capture["bev_encoder.ffn"])
    assert bev_cross_attn_out.shape == (expected_bev, batch_size, expected_c)
    assert bev_ffn_out.shape == (expected_bev, batch_size, expected_c)

    for idx in range(expected_l):
        layer_out = _first_tensor(capture[f"decoder.layer{idx}"])
        self_attn_out = _first_tensor(capture[f"decoder.layer{idx}.self_attn"])
        cross_attn_out = _first_tensor(capture[f"decoder.layer{idx}.cross_attn"])
        ffn_out = _first_tensor(capture[f"decoder.layer{idx}.ffn"])
        assert layer_out.shape == (expected_q, batch_size, expected_c)
        assert self_attn_out.shape == (expected_q, batch_size, expected_c)
        assert cross_attn_out.shape == (expected_q, batch_size, expected_c)
        assert ffn_out.shape == (expected_q, batch_size, expected_c)
        assert _first_tensor(capture[f"head.cls_branch{idx}"]).shape == (batch_size, expected_v, cfg.num_map_classes)
        assert _first_tensor(capture[f"head.reg_branch{idx}"]).shape == (batch_size, expected_q, cfg.code_size)

    assert outputs["bev_embed"].shape == (batch_size, expected_bev, expected_c)
    assert outputs["all_cls_scores"].shape == (expected_l, batch_size, expected_v, cfg.num_map_classes)
    assert outputs["all_bbox_preds"].shape == (expected_l, batch_size, expected_v, 4)
    assert outputs["all_pts_preds"].shape == (expected_l, batch_size, expected_v, expected_p, 2)


def test_metadata_contract_validation_requires_keys_and_shapes():
    cfg = debug_forward_config(num_vec=10, num_pts_per_vec=4, decoder_layers=2)
    model = MapTRLite(cfg).eval()
    batch_size = 1
    height, width = 96, 160
    img = torch.randn(batch_size, cfg.num_cams, 3, height, width)
    img_metas = _build_dummy_img_metas(batch_size=batch_size, num_cams=cfg.num_cams, height=height, width=width)

    with torch.no_grad():
        outputs = model(img, img_metas, decode=False)
    assert isinstance(outputs, dict)

    missing_pad_shape = copy.deepcopy(img_metas)
    missing_pad_shape[0].pop("pad_shape")
    with pytest.raises(KeyError, match="pad_shape"):
        model(img, missing_pad_shape, decode=False)

    bad_img_shape = copy.deepcopy(img_metas)
    bad_img_shape[0]["img_shape"] = bad_img_shape[0]["img_shape"][:-1]
    with pytest.raises(ValueError, match="img_shape"):
        model(img, bad_img_shape, decode=False)

    bad_pad_shape = copy.deepcopy(img_metas)
    bad_pad_shape[0]["pad_shape"] = [(-1, width, 3) for _ in range(cfg.num_cams)]
    with pytest.raises(ValueError, match="pad_shape"):
        model(img, bad_pad_shape, decode=False)


def test_decode_contract_semantics_and_topk_consistency(instrumented_debug_forward):
    data = instrumented_debug_forward
    cfg = data["cfg"]
    model = data["model"]

    with torch.no_grad():
        decoded_pack = model(data["img"], data["img_metas"], decode=True)

    assert isinstance(decoded_pack, dict)
    assert set(decoded_pack.keys()) == {"preds", "decoded"}
    preds = decoded_pack["preds"]
    decoded = decoded_pack["decoded"]
    assert isinstance(preds, dict)
    assert isinstance(decoded, list)
    assert len(decoded) == data["batch_size"]

    cls_last = preds["all_cls_scores"][-1]
    for batch_idx, sample in enumerate(decoded):
        assert set(sample.keys()) == {"bboxes", "pts", "scores", "labels"}
        assert sample["bboxes"].ndim == 2
        assert sample["bboxes"].shape[1] == 4
        assert sample["pts"].ndim == 3
        assert sample["pts"].shape[1:] == (cfg.num_pts_per_vec, 2)
        assert sample["scores"].ndim == 1
        assert sample["labels"].ndim == 1
        assert sample["scores"].shape == sample["labels"].shape
        assert sample["scores"].shape[0] == sample["bboxes"].shape[0] == sample["pts"].shape[0]
        assert sample["labels"].dtype == torch.long
        _assert_decoded_topk_label_score_consistency(
            cls_last[batch_idx],
            sample["scores"],
            sample["labels"],
            max_num=cfg.max_num,
            num_classes=cfg.num_map_classes,
        )


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
        if value is None:
            continue
        tensors = list(_iter_tensors(value))
        assert tensors, f"No tensor found in output '{name}'."
        for tensor in tensors:
            assert torch.isfinite(tensor).all(), f"Non-finite values found in final output '{name}'."
