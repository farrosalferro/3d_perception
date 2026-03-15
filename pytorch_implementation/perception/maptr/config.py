"""Configuration objects for the pure PyTorch MapTR forward path."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple


@dataclass(frozen=True)
class BackboneNeckConfig:
    """Simple backbone/FPN config used by the standalone MapTR model."""

    stage_channels: Tuple[int, int, int, int] = (64, 128, 256, 512)
    out_indices: Tuple[int, ...] = (3,)
    out_channels: int = 256
    num_outs: int = 1


@dataclass(frozen=True)
class MapTRForwardConfig:
    """Forward-only MapTR configuration."""

    name: str
    num_map_classes: int = 3
    num_vec: int = 50
    num_pts_per_vec: int = 20
    num_cams: int = 6
    embed_dims: int = 256
    ffn_dims: int = 512
    num_heads: int = 8
    num_decoder_layers: int = 6
    bev_h: int = 30
    bev_w: int = 30
    query_embed_type: str = "instance_pts"
    with_box_refine: bool = True
    code_size: int = 2
    pc_range: Tuple[float, float, float, float, float, float] = (
        -15.0,
        -30.0,
        -2.0,
        15.0,
        30.0,
        2.0,
    )
    # MapTR coder contract for xyxy decode filtering:
    # [x_min, y_min, x_min, y_min, x_max, y_max, x_max, y_max].
    post_center_range: Tuple[float, float, float, float, float, float, float, float] = (
        -20.0,
        -35.0,
        -20.0,
        -35.0,
        20.0,
        35.0,
        20.0,
        35.0,
    )
    max_num: int = 50
    score_threshold: float | None = None
    dropout: float = 0.1
    strict_img_meta: bool = True
    backbone_neck: BackboneNeckConfig = BackboneNeckConfig()

    @property
    def num_query(self) -> int:
        return int(self.num_vec * self.num_pts_per_vec)


def maptr_tiny_forward_config() -> MapTRForwardConfig:
    """Config aligned with MapTR tiny-style defaults."""

    return MapTRForwardConfig(name="maptr_tiny_forward")


def debug_forward_config(
    *,
    num_vec: int = 12,
    num_pts_per_vec: int = 6,
    decoder_layers: int = 2,
) -> MapTRForwardConfig:
    """Small config for fast local smoke tests."""

    base = maptr_tiny_forward_config()
    return replace(
        base,
        name=f"{base.name}_debug",
        num_vec=int(num_vec),
        num_pts_per_vec=int(num_pts_per_vec),
        num_decoder_layers=int(decoder_layers),
        embed_dims=128,
        ffn_dims=256,
        max_num=min(100, int(num_vec)),
        backbone_neck=replace(
            base.backbone_neck,
            stage_channels=(32, 64, 96, 128),
            out_channels=128,
        ),
    )


def replace_backbone(cfg: MapTRForwardConfig, **kwargs: object) -> MapTRForwardConfig:
    """Return cfg with a replaced BackboneNeckConfig."""

    return replace(cfg, backbone_neck=replace(cfg.backbone_neck, **kwargs))


def replace_config(cfg: MapTRForwardConfig, **kwargs: object) -> MapTRForwardConfig:
    """Return cfg with selected values replaced."""

    return replace(cfg, **kwargs)
