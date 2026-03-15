"""Intermediate tensor validation tests for pure-PyTorch FB-BEV."""

from __future__ import annotations

import copy
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from pytorch_implementation.perception.fbbev.config import debug_forward_config
from pytorch_implementation.perception.fbbev.model import FBBEVLite
from tests._shared.hook_helpers import register_hook_overwrite
from tests._shared.parity_helpers import assert_decoded_topk_label_score_consistency
from tests._shared.tensor_helpers import conv2d_out, first_tensor, iter_tensors


def _build_dummy_img_metas(batch_size: int) -> list[dict[str, Any]]:
    metas = []
    for batch_idx in range(batch_size):
        curr_to_prev = torch.eye(4, dtype=torch.float32)
        curr_to_prev[0, 3] = 0.4 * (batch_idx + 1)
        curr_to_prev[1, 3] = -0.2 * (batch_idx + 1)
        metas.append(
            {
                "sample_idx": f"sample_{batch_idx}",
                "sequence_group_idx": batch_idx,
                "start_of_sequence": True,
                "curr_to_prev_ego_rt": curr_to_prev,
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
    cfg = debug_forward_config(max_num=48, depth_bins=6, bev_h=20, bev_w=20, bev_z=3)
    model = FBBEVLite(cfg).eval()

    batch_size = 2
    height, width = 96, 160
    img = torch.randn(batch_size, cfg.num_cams, 3, height, width)
    img_metas = _build_dummy_img_metas(batch_size=batch_size)

    capture: dict[str, Any] = {}
    handles = []

    # Backbone/FPN hooks.
    _register_hook(model.backbone_neck.backbone.stem, "backbone.stem", capture, handles)
    for idx, stage in enumerate(model.backbone_neck.backbone.stages):
        _register_hook(stage, f"backbone.stage{idx}", capture, handles)
    _register_hook(model.backbone_neck.neck.output_convs[0], "fpn.output0", capture, handles)
    _register_hook(model.backbone_neck.neck.output_convs[1], "fpn.output1", capture, handles)

    # Core FB-BEV pipeline hooks.
    _register_hook(model.depth_net.trunk, "depth_net.trunk", capture, handles)
    _register_hook(model.depth_net.context_proj, "depth_net.context_proj", capture, handles)
    _register_hook(model.depth_net.depth_logits, "depth_net.depth_logits", capture, handles)
    _register_hook(model.forward_projection, "forward_projection", capture, handles)
    _register_hook(model.backward_projection.depth_attention, "backward.depth_attention", capture, handles)
    _register_hook(model.backward_projection.post, "backward.post", capture, handles)
    _register_hook(model.temporal_fusion.time_conv, "temporal.time_conv", capture, handles)
    _register_hook(model.temporal_fusion.cat_conv, "temporal.cat_conv", capture, handles)
    _register_hook(model.bev_encoder.blocks[0], "bev_encoder.block0", capture, handles)
    _register_hook(model.bev_encoder.blocks[1], "bev_encoder.block1", capture, handles)
    _register_hook(model.detection_head.shared, "head.shared", capture, handles)
    _register_hook(model.detection_head.heatmap_head, "head.heatmap", capture, handles)
    _register_hook(model.detection_head.reg_head, "head.reg", capture, handles)

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
    capture = data["capture"]

    expected_names = {
        "backbone.stem",
        "backbone.stage0",
        "backbone.stage1",
        "backbone.stage2",
        "backbone.stage3",
        "fpn.output0",
        "fpn.output1",
        "depth_net.trunk",
        "depth_net.context_proj",
        "depth_net.depth_logits",
        "forward_projection",
        "backward.depth_attention",
        "backward.post",
        "temporal.time_conv",
        "temporal.cat_conv",
        "bev_encoder.block0",
        "bev_encoder.block1",
        "head.shared",
        "head.heatmap",
        "head.reg",
    }
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
    assert _first_tensor(capture["fpn.output0"]).shape == (batch_size * cams, cfg.backbone_neck.out_channels, stage2_h, stage2_w)
    assert _first_tensor(capture["fpn.output1"]).shape == (batch_size * cams, cfg.backbone_neck.out_channels, stage3_h, stage3_w)

    assert _first_tensor(capture["depth_net.trunk"]).shape == (
        batch_size * cams,
        cfg.backbone_neck.out_channels,
        stage2_h,
        stage2_w,
    )
    assert _first_tensor(capture["depth_net.context_proj"]).shape == (batch_size * cams, cfg.embed_dims, stage2_h, stage2_w)
    assert _first_tensor(capture["depth_net.depth_logits"]).shape == (batch_size * cams, cfg.depth_bins, stage2_h, stage2_w)
    assert _first_tensor(capture["forward_projection"]).shape == (
        batch_size,
        cfg.embed_dims,
        cfg.bev_h,
        cfg.bev_w,
        cfg.bev_z,
    )

    expected_2d = (batch_size, cfg.embed_dims, cfg.bev_h, cfg.bev_w)
    assert _first_tensor(capture["backward.depth_attention"]).shape == expected_2d
    assert _first_tensor(capture["backward.post"]).shape == expected_2d
    assert _first_tensor(capture["temporal.time_conv"]).shape == (
        batch_size * (cfg.history_cat_num + 1),
        cfg.embed_dims,
        cfg.bev_h,
        cfg.bev_w,
    )
    assert _first_tensor(capture["temporal.cat_conv"]).shape == expected_2d
    assert _first_tensor(capture["bev_encoder.block0"]).shape == expected_2d
    assert _first_tensor(capture["bev_encoder.block1"]).shape == expected_2d
    assert _first_tensor(capture["head.shared"]).shape == expected_2d
    assert _first_tensor(capture["head.heatmap"]).shape == (batch_size, cfg.num_classes, cfg.bev_h, cfg.bev_w)
    assert _first_tensor(capture["head.reg"]).shape == (batch_size, cfg.code_size, cfg.bev_h, cfg.bev_w)

    assert outputs["all_cls_scores"].shape == (1, batch_size, cfg.max_num, cfg.num_classes)
    assert outputs["all_bbox_preds"].shape == (1, batch_size, cfg.max_num, cfg.code_size)
    assert outputs["context"].shape == (batch_size, cfg.num_cams, cfg.embed_dims, stage2_h, stage2_w)
    assert outputs["depth"].shape == (batch_size, cfg.num_cams, cfg.depth_bins, stage2_h, stage2_w)
    assert outputs["bev_volume"].shape == (batch_size, cfg.embed_dims, cfg.bev_h, cfg.bev_w, cfg.bev_z)
    assert outputs["bev_refined"].shape == expected_2d
    assert outputs["bev_fused"].shape == expected_2d
    assert outputs["bev_embed"].shape == expected_2d


def test_metadata_contract_validation_requires_temporal_and_shape_keys():
    cfg = debug_forward_config(max_num=24, depth_bins=6, bev_h=20, bev_w=20, bev_z=3)
    model = FBBEVLite(cfg).eval()
    batch_size = 2
    height, width = 96, 160
    img = torch.randn(batch_size, cfg.num_cams, 3, height, width)
    img_metas = _build_dummy_img_metas(batch_size=batch_size)

    with torch.no_grad():
        outputs = model(img, img_metas, decode=False)
    assert isinstance(outputs, dict)

    missing_temporal_key = copy.deepcopy(img_metas)
    missing_temporal_key[0].pop("sequence_group_idx")
    with pytest.raises(KeyError, match="sequence_group_idx"):
        model(img, missing_temporal_key, decode=False)

    bad_curr_to_prev_shape = copy.deepcopy(img_metas)
    bad_curr_to_prev_shape[0]["curr_to_prev_ego_rt"] = torch.eye(3, dtype=torch.float32)
    with pytest.raises(ValueError, match="curr_to_prev_ego_rt"):
        model(img, bad_curr_to_prev_shape, decode=False)

    bad_curr_to_prev_values = copy.deepcopy(img_metas)
    bad_curr_to_prev_values[0]["curr_to_prev_ego_rt"] = torch.full((4, 4), float("nan"))
    with pytest.raises(ValueError, match="finite"):
        model(img, bad_curr_to_prev_values, decode=False)


def test_decode_contract_semantics_and_topk_consistency(instrumented_debug_forward):
    data = instrumented_debug_forward
    cfg = data["cfg"]
    model = data["model"]

    with torch.no_grad():
        decoded_pack = model(data["img"], data["img_metas"], decode=True)

    assert isinstance(decoded_pack, dict)
    assert set(decoded_pack.keys()) == {"preds", "decoded", "decoded_occupancy"}
    preds = decoded_pack["preds"]
    decoded = decoded_pack["decoded"]
    decoded_occ = decoded_pack["decoded_occupancy"]
    assert isinstance(preds, dict)
    assert isinstance(decoded, list)
    assert isinstance(decoded_occ, list)
    assert len(decoded) == data["batch_size"]
    assert len(decoded_occ) == data["batch_size"]

    cls_last = preds["all_cls_scores"][-1]
    for batch_idx, sample in enumerate(decoded):
        assert set(sample.keys()) == {"bboxes", "scores", "labels"}
        assert sample["bboxes"].ndim == 2
        assert sample["bboxes"].shape[1] == cfg.code_size
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

    for occ in decoded_occ:
        assert occ.shape == (cfg.bev_h, cfg.bev_w, cfg.bev_z)
        assert occ.dtype == torch.long
        if occ.numel() > 0:
            assert int(occ.min().item()) >= 0
            assert int(occ.max().item()) < cfg.occupancy_classes


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
