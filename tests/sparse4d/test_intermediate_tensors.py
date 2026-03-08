"""Intermediate tensor validation tests for pure-PyTorch Sparse4D."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from pytorch_implementation.sparse4d.config import debug_forward_config
from pytorch_implementation.sparse4d.model import Sparse4DLite


def _build_dummy_metas(batch_size: int, num_cams: int, height: int, width: int) -> dict[str, torch.Tensor]:
    projection_mat = torch.eye(4, dtype=torch.float32).view(1, 1, 4, 4).repeat(batch_size, num_cams, 1, 1)
    for cam_idx in range(num_cams):
        projection_mat[:, cam_idx, 0, 3] = width * (0.45 + 0.02 * cam_idx)
        projection_mat[:, cam_idx, 1, 3] = height * 0.5
    image_wh = torch.tensor([width, height], dtype=torch.float32).view(1, 1, 2).repeat(batch_size, num_cams, 1)
    return {"projection_mat": projection_mat, "image_wh": image_wh}


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
    cfg = debug_forward_config(num_queries=48, decoder_layers=2)
    model = Sparse4DLite(cfg).eval()

    batch_size = 1
    height, width = 96, 160
    img = torch.randn(batch_size, cfg.num_cams, 3, height, width)
    metas = _build_dummy_metas(batch_size=batch_size, num_cams=cfg.num_cams, height=height, width=width)

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

    for handle in handles:
        handle.remove()

    assert isinstance(outputs, dict)
    return {
        "cfg": cfg,
        "model": model,
        "img": img,
        "metas": metas,
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
