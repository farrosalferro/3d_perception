"""Configuration objects for the pure PyTorch PETR forward path."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence, Tuple


@dataclass(frozen=True)
class BackboneNeckConfig:
    """Simple backbone/FPN config used by the standalone PETR model."""

    stage_channels: Tuple[int, int, int, int] = (64, 128, 256, 512)
    out_indices: Tuple[int, ...] = (3,)
    out_channels: int = 256
    num_outs: int = 1


@dataclass(frozen=True)
class PETRForwardConfig:
    """Forward-only PETR configuration."""

    name: str
    num_classes: int = 10
    num_queries: int = 900
    num_cams: int = 6
    embed_dims: int = 256
    ffn_dims: int = 512
    num_heads: int = 8
    num_decoder_layers: int = 6
    code_size: int = 10
    depth_num: int = 16
    depth_start: float = 1.0
    lidar_discretization: bool = False
    position_range: Tuple[float, float, float, float, float, float] = (
        -61.2,
        -61.2,
        -10.0,
        61.2,
        61.2,
        10.0,
    )
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


def petr_r50_forward_config() -> PETRForwardConfig:
    """Config aligned with PETR-R50 style defaults."""

    return PETRForwardConfig(name="petr_r50_forward")


def debug_forward_config(
    *,
    num_queries: int = 64,
    decoder_layers: int = 3,
    depth_num: int = 8,
) -> PETRForwardConfig:
    """Small config for fast local smoke tests."""

    base = petr_r50_forward_config()
    return replace(
        base,
        name=f"{base.name}_debug",
        num_queries=int(num_queries),
        num_decoder_layers=int(decoder_layers),
        depth_num=int(depth_num),
        max_num=min(100, int(num_queries)),
    )


def replace_backbone(cfg: PETRForwardConfig, **kwargs: object) -> PETRForwardConfig:
    """Return cfg with a replaced BackboneNeckConfig."""

    return replace(cfg, backbone_neck=replace(cfg.backbone_neck, **kwargs))


def replace_config(cfg: PETRForwardConfig, **kwargs: object) -> PETRForwardConfig:
    """Return cfg with selected values replaced."""

    return replace(cfg, **kwargs)

