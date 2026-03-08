"""Configuration objects for the pure PyTorch FB-BEV forward path."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple


@dataclass(frozen=True)
class BackboneNeckConfig:
    """Simple backbone/FPN config used by the standalone FB-BEV model."""

    stage_channels: Tuple[int, int, int, int] = (48, 96, 160, 256)
    out_indices: Tuple[int, ...] = (2, 3)
    out_channels: int = 128
    num_outs: int = 2


@dataclass(frozen=True)
class FBBEVForwardConfig:
    """Forward-only FB-BEV configuration."""

    name: str
    num_classes: int = 10
    num_cams: int = 6
    embed_dims: int = 128
    depth_bins: int = 8
    bev_h: int = 32
    bev_w: int = 32
    bev_z: int = 4
    code_size: int = 9
    history_cat_num: int = 2
    history_cam_sweep_freq: float = 0.5
    use_temporal_fusion: bool = True
    max_num: int = 120
    score_threshold: float | None = None
    voxel_size: Tuple[float, float] = (0.8, 0.8)
    pc_range: Tuple[float, float, float, float, float, float] = (
        -51.2,
        -51.2,
        -5.0,
        51.2,
        51.2,
        3.0,
    )
    post_center_range: Tuple[float, float, float, float, float, float] = (
        -61.2,
        -61.2,
        -10.0,
        61.2,
        61.2,
        10.0,
    )
    backbone_neck: BackboneNeckConfig = BackboneNeckConfig()


def fbbev_base_forward_config() -> FBBEVForwardConfig:
    """Config aligned with a compact FB-BEV-style setup."""

    return FBBEVForwardConfig(name="fbbev_base_forward")


def debug_forward_config(
    *,
    num_cams: int = 6,
    depth_bins: int = 6,
    bev_h: int = 20,
    bev_w: int = 20,
    bev_z: int = 3,
    history_cat_num: int = 2,
    max_num: int = 64,
) -> FBBEVForwardConfig:
    """Small config for fast local smoke tests."""

    base = fbbev_base_forward_config()
    return replace(
        base,
        name=f"{base.name}_debug",
        num_cams=int(num_cams),
        depth_bins=int(depth_bins),
        bev_h=int(bev_h),
        bev_w=int(bev_w),
        bev_z=int(bev_z),
        history_cat_num=int(history_cat_num),
        max_num=int(max_num),
    )


def replace_backbone(cfg: FBBEVForwardConfig, **kwargs: object) -> FBBEVForwardConfig:
    """Return cfg with a replaced BackboneNeckConfig."""

    return replace(cfg, backbone_neck=replace(cfg.backbone_neck, **kwargs))


def replace_config(cfg: FBBEVForwardConfig, **kwargs: object) -> FBBEVForwardConfig:
    """Return cfg with selected values replaced."""

    return replace(cfg, **kwargs)
