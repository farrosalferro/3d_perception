"""Intermediate tensor validation tests for pure-PyTorch PolarFormer."""

from __future__ import annotations

import copy
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from pytorch_implementation.perception.polarformer.config import debug_forward_config
from pytorch_implementation.perception.polarformer.model import PolarFormerLite
from tests._shared.hook_helpers import register_hook_overwrite
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
        cam_intrinsic = []
        cam2lidar = []
        for cam_idx in range(num_cams):
            projection = [
                [1.0, 0.0, 0.0, float(width) * (0.5 + cam_idx * 0.01)],
                [0.0, 1.0, 0.0, float(height) * 0.5],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            lidar2img.append(projection)
            cam_intrinsic.append(
                [
                    [float(width), 0.0, float(width) * 0.5],
                    [0.0, float(height), float(height) * 0.5],
                    [0.0, 0.0, 1.0],
                ]
            )
            cam2lidar.append(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        metas.append(
            {
                "sample_idx": f"sample_{batch_idx}",
                "lidar2img": lidar2img,
                "cam_intrinsic": cam_intrinsic,
                "cam2lidar": cam2lidar,
                "img_shape": [(height, width, 3) for _ in range(num_cams)],
                "pad_shape": [(height, width, 3) for _ in range(num_cams)],
            }
        )
    return metas


def _first_tensor(value: Any) -> torch.Tensor | None:
    return first_tensor(value)


def _iter_tensors(value: Any):
    yield from iter_tensors(value)


def _register_hook(module, name: str, capture: dict[str, Any], handles: list) -> None:
    register_hook_overwrite(module, name, capture, handles)


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
    cfg = debug_forward_config(num_queries=48, decoder_layers=2, azimuth_bins=96, radius_bins=48)
    model = PolarFormerLite(cfg).eval()

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
    for idx, conv in enumerate(model.backbone_neck.fpn.output_convs):
        _register_hook(conv, f"fpn.output{idx}", capture, handles)

    # Polar neck hooks.
    for idx, projector in enumerate(model.backbone_neck.polar_projectors):
        _register_hook(projector, f"polar.projector{idx}", capture, handles)
    for idx, conv in enumerate(model.backbone_neck.polar_out_convs):
        _register_hook(conv, f"polar.output{idx}", capture, handles)

    # Head + decoder hooks.
    for idx, proj in enumerate(model.head.input_projs):
        _register_hook(proj, f"head.input_proj{idx}", capture, handles)
    _register_hook(model.head.query_embedding, "head.query_embedding", capture, handles)
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
        "fpn.output1",
        "fpn.output2",
        "polar.projector0",
        "polar.projector1",
        "polar.projector2",
        "polar.output0",
        "polar.output1",
        "polar.output2",
        "head.input_proj0",
        "head.input_proj1",
        "head.input_proj2",
        "head.query_embedding",
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

    assert _first_tensor(capture["fpn.output0"]).shape == (batch_size * cams, cfg.backbone_neck.out_channels, stage1_h, stage1_w)
    assert _first_tensor(capture["fpn.output1"]).shape == (batch_size * cams, cfg.backbone_neck.out_channels, stage2_h, stage2_w)
    assert _first_tensor(capture["fpn.output2"]).shape == (batch_size * cams, cfg.backbone_neck.out_channels, stage3_h, stage3_w)

    azimuth_bins, radius_bins, _ = cfg.polar_neck.output_size
    expected_sizes = [
        (radius_bins, azimuth_bins),
        (radius_bins // 2, azimuth_bins // 2),
        (radius_bins // 4, azimuth_bins // 4),
    ]
    expected_widths = [stage1_w, stage2_w, stage3_w]
    for lvl, (target_r, target_a) in enumerate(expected_sizes):
        projector = _first_tensor(capture[f"polar.projector{lvl}"])
        assert projector.shape == (target_r, batch_size * expected_widths[lvl], cfg.embed_dims)
        assert _first_tensor(capture[f"polar.output{lvl}"]).shape == (batch_size, cfg.embed_dims, target_r, target_a)
        assert _first_tensor(capture[f"head.input_proj{lvl}"]).shape == (batch_size, cfg.embed_dims, target_r, target_a)

    assert _first_tensor(capture["head.query_embedding"]).shape == (cfg.num_queries, cfg.embed_dims * 2)
    for idx in range(cfg.num_decoder_layers):
        assert _first_tensor(capture[f"decoder.layer{idx}"]).shape == (cfg.num_queries, batch_size, cfg.embed_dims)
        assert _first_tensor(capture[f"decoder.layer{idx}.self_attn"]).shape == (cfg.num_queries, batch_size, cfg.embed_dims)
        assert _first_tensor(capture[f"decoder.layer{idx}.cross_attn"]).shape == (cfg.num_queries, batch_size, cfg.embed_dims)
        assert _first_tensor(capture[f"decoder.layer{idx}.ffn"]).shape == (cfg.num_queries, batch_size, cfg.embed_dims)
        assert _first_tensor(capture[f"head.cls_branch{idx}"]).shape == (batch_size, cfg.num_queries, cfg.num_classes)
        assert _first_tensor(capture[f"head.reg_branch{idx}"]).shape == (batch_size, cfg.num_queries, cfg.code_size)

    assert outputs["all_cls_scores"].shape == (
        cfg.num_decoder_layers,
        batch_size,
        cfg.num_queries,
        cfg.num_classes,
    )
    assert outputs["all_bbox_preds"].shape == (
        cfg.num_decoder_layers,
        batch_size,
        cfg.num_queries,
        cfg.code_size,
    )


def test_metadata_contract_validation_requires_keys_and_shapes():
    cfg = debug_forward_config(num_queries=24, decoder_layers=2, azimuth_bins=96, radius_bins=48)
    model = PolarFormerLite(cfg).eval()
    batch_size = 1
    height, width = 96, 160
    img = torch.randn(batch_size, cfg.num_cams, 3, height, width)
    img_metas = _build_dummy_img_metas(batch_size=batch_size, num_cams=cfg.num_cams, height=height, width=width)

    with torch.no_grad():
        outputs = model(img, img_metas, decode=False)
    assert isinstance(outputs, dict)

    missing_geometry = copy.deepcopy(img_metas)
    missing_geometry[0].pop("cam_intrinsic")
    with pytest.raises(KeyError, match="cam_intrinsic"):
        model(img, missing_geometry, decode=False)

    bad_lidar2img = copy.deepcopy(img_metas)
    bad_lidar2img[0]["lidar2img"] = bad_lidar2img[0]["lidar2img"][:-1]
    with pytest.raises(ValueError, match="lidar2img"):
        model(img, bad_lidar2img, decode=False)

    bad_pad_shape = copy.deepcopy(img_metas)
    bad_pad_shape[0]["pad_shape"] = [(height - 8, width - 8, 3) for _ in range(cfg.num_cams)]
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

