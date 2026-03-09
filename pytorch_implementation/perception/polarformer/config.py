"""Configuration objects for the pure PyTorch PolarFormer forward path."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple


@dataclass(frozen=True)
class BackboneNeckConfig:
    """Backbone/FPN config for PolarFormer-lite image features."""

    stage_channels: Tuple[int, int, int, int] = (64, 128, 256, 512)
    out_indices: Tuple[int, ...] = (1, 2, 3)
    out_channels: int = 256
    num_outs: int = 3


@dataclass(frozen=True)
class PolarNeckConfig:
    """Polar projection config for camera-to-polar BEV features."""

    num_levels: int = 3
    num_heads: int = 8
    radius_range: Tuple[float, float, float] = (1.0, 65.0, 1.0)
    output_size: Tuple[int, int, int] = (128, 64, 10)  # [azimuth, radius, height]
    use_different_res: bool = True


@dataclass(frozen=True)
class PolarFormerForwardConfig:
    """Forward-only PolarFormer configuration."""

    name: str
    num_classes: int = 10
    num_queries: int = 900
    num_cams: int = 6
    embed_dims: int = 256
    ffn_dims: int = 512
    num_heads: int = 8
    num_decoder_layers: int = 6
    code_size: int = 10
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
    max_num: int = 300
    score_threshold: float | None = None
    dropout: float = 0.1
    backbone_neck: BackboneNeckConfig = BackboneNeckConfig()
    polar_neck: PolarNeckConfig = PolarNeckConfig()


def polarformer_r101_forward_config() -> PolarFormerForwardConfig:
    """Config aligned with PolarFormer-R101 style defaults."""

    return PolarFormerForwardConfig(name="polarformer_r101_forward")


def debug_forward_config(
    *,
    num_queries: int = 48,
    decoder_layers: int = 2,
    azimuth_bins: int = 96,
    radius_bins: int = 48,
) -> PolarFormerForwardConfig:
    """Small config for fast local smoke tests."""

    base = polarformer_r101_forward_config()
    return replace(
        base,
        name=f"{base.name}_debug",
        num_queries=int(num_queries),
        num_decoder_layers=int(decoder_layers),
        max_num=min(100, int(num_queries)),
        dropout=0.0,
        polar_neck=replace(base.polar_neck, output_size=(int(azimuth_bins), int(radius_bins), 10)),
    )

