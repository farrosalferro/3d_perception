"""Intermediate tensor validation tests for pure-PyTorch BEVFormer."""

from __future__ import annotations

import math
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from pure_torch_bevformer.config import debug_forward_config
from pure_torch_bevformer.model import BEVFormerLite


def _build_dummy_img_metas(
    batch_size: int,
    num_cams: int,
    height: int,
    width: int,
    *,
    can_bus_dims: int = 18,
) -> list[dict[str, Any]]:
    metas = []
    for batch_idx in range(batch_size):
        can_bus = [0.0] * can_bus_dims
        lidar2img = []
        for cam_idx in range(num_cams):
            # Keep projected coordinates in-frame for most BEV points:
            # x_img ~= x_world + width/2, y_img ~= y_world + height/2, z~=1.
            projection = [
                [1.0, 0.0, 0.0, float(width) * (0.5 + cam_idx * 0.005)],
                [0.0, 1.0, 0.0, float(height) * 0.5],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            lidar2img.append(projection)
        metas.append(
            {
                "scene_token": f"scene_{batch_idx}",
                "can_bus": can_bus,
                "lidar2img": lidar2img,
                "img_shape": [(height, width, 3) for _ in range(num_cams)],
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


def _conv2d_out(size: int, kernel: int, stride: int, padding: int) -> int:
    return ((size + 2 * padding - kernel) // stride) + 1


@pytest.fixture()
def instrumented_debug_forward():
    cfg = debug_forward_config("tiny", bev_hw=(12, 12), num_queries=48, encoder_layers=2, decoder_layers=2)
    model = BEVFormerLite(cfg).eval()

    batch_size = 1
    height, width = 96, 160
    img = torch.randn(batch_size, cfg.num_cams, 3, height, width)
    img_metas = _build_dummy_img_metas(
        batch_size=batch_size,
        num_cams=cfg.num_cams,
        height=height,
        width=width,
        can_bus_dims=cfg.can_bus_dims,
    )

    capture: dict[str, Any] = {}
    handles = []

    # Backbone/FPN hooks.
    _register_hook(model.backbone_neck.backbone.stem, "backbone.stem", capture, handles)
    for idx, stage in enumerate(model.backbone_neck.backbone.stages):
        _register_hook(stage, f"backbone.stage{idx}", capture, handles)
    for idx, lateral in enumerate(model.backbone_neck.neck.lateral_convs):
        _register_hook(lateral, f"fpn.lateral{idx}", capture, handles)
    for idx, output_conv in enumerate(model.backbone_neck.neck.output_convs):
        _register_hook(output_conv, f"fpn.output{idx}", capture, handles)

    # Head/transformer hooks.
    _register_hook(model.head.positional_encoding, "head.positional_encoding", capture, handles)
    _register_hook(model.head.transformer.can_bus_mlp, "transformer.can_bus_mlp", capture, handles)
    _register_hook(model.head.transformer.reference_points, "transformer.reference_points_linear", capture, handles)

    for idx, layer in enumerate(model.head.transformer.encoder.layers):
        _register_hook(layer, f"encoder.layer{idx}", capture, handles)
        _register_hook(layer.temporal_attn, f"encoder.layer{idx}.temporal", capture, handles)
        _register_hook(layer.spatial_attn, f"encoder.layer{idx}.spatial", capture, handles)
        _register_hook(layer.spatial_attn.deformable_attention, f"encoder.layer{idx}.spatial_deformable", capture, handles)
        _register_hook(layer.ffn, f"encoder.layer{idx}.ffn", capture, handles)

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
        "fpn.lateral0",
        "fpn.output0",
        "head.positional_encoding",
        "transformer.can_bus_mlp",
        "transformer.reference_points_linear",
    }
    for idx in range(cfg.num_encoder_layers):
        expected_names.update(
            {
                f"encoder.layer{idx}",
                f"encoder.layer{idx}.temporal",
                f"encoder.layer{idx}.spatial",
                f"encoder.layer{idx}.spatial_deformable",
                f"encoder.layer{idx}.ffn",
            }
        )
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

    expected_hw = cfg.bev_h * cfg.bev_w
    expected_q = cfg.num_queries
    expected_c = cfg.embed_dims

    assert _first_tensor(capture["head.positional_encoding"]).shape == (batch_size, expected_c, cfg.bev_h, cfg.bev_w)
    assert _first_tensor(capture["transformer.can_bus_mlp"]).shape == (batch_size, expected_c)
    assert _first_tensor(capture["transformer.reference_points_linear"]).shape == (batch_size, expected_q, 3)

    for idx in range(cfg.num_encoder_layers):
        assert _first_tensor(capture[f"encoder.layer{idx}"]).shape == (batch_size, expected_hw, expected_c)
        assert _first_tensor(capture[f"encoder.layer{idx}.temporal"]).shape == (batch_size, expected_hw, expected_c)
        assert _first_tensor(capture[f"encoder.layer{idx}.spatial"]).shape == (batch_size, expected_hw, expected_c)
        assert _first_tensor(capture[f"encoder.layer{idx}.ffn"]).shape == (batch_size, expected_hw, expected_c)

        spatial_deform = _first_tensor(capture[f"encoder.layer{idx}.spatial_deformable"])
        assert spatial_deform.shape[0] == batch_size * cams
        assert spatial_deform.shape[-1] == expected_c

    for idx in range(cfg.num_decoder_layers):
        self_attn_tensor = _first_tensor(capture[f"decoder.layer{idx}.self_attn"])
        assert self_attn_tensor.shape == (expected_q, batch_size, expected_c)
        assert _first_tensor(capture[f"decoder.layer{idx}.cross_attn"]).shape == (expected_q, batch_size, expected_c)
        assert _first_tensor(capture[f"decoder.layer{idx}.ffn"]).shape == (expected_q, batch_size, expected_c)
        assert _first_tensor(capture[f"decoder.layer{idx}"]).shape == (expected_q, batch_size, expected_c)
        assert _first_tensor(capture[f"head.cls_branch{idx}"]).shape == (batch_size, expected_q, cfg.num_classes)
        assert _first_tensor(capture[f"head.reg_branch{idx}"]).shape == (batch_size, expected_q, 10)

    assert outputs["bev_embed"].shape == (expected_hw, batch_size, expected_c)
    assert outputs["all_cls_scores"].shape == (cfg.num_decoder_layers, batch_size, expected_q, cfg.num_classes)
    assert outputs["all_bbox_preds"].shape == (cfg.num_decoder_layers, batch_size, expected_q, 10)


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
