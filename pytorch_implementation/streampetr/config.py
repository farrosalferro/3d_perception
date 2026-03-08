"""Configuration objects for the pure PyTorch StreamPETR forward path."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple


@dataclass(frozen=True)
class BackboneNeckConfig:
    """Simple backbone/FPN config used by standalone StreamPETR."""

    stage_channels: Tuple[int, int, int, int] = (64, 128, 256, 512)
    out_indices: Tuple[int, ...] = (3,)
    out_channels: int = 256
    num_outs: int = 1


@dataclass(frozen=True)
class StreamPETRForwardConfig:
    """Forward-only StreamPETR configuration."""

    name: str
    num_classes: int = 10
    num_queries: int = 644
    num_cams: int = 6
    embed_dims: int = 256
    ffn_dims: int = 512
    num_heads: int = 8
    num_decoder_layers: int = 6
    code_size: int = 10
    depth_num: int = 16
    depth_start: float = 1.0
    lidar_discretization: bool = True
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
    memory_len: int = 1024
    topk_proposals: int = 256
    num_propagated: int = 256
    with_ego_pos: bool = True
    backbone_neck: BackboneNeckConfig = BackboneNeckConfig()


def streampetr_r50_forward_config() -> StreamPETRForwardConfig:
    """Config aligned with StreamPETR-R50 style defaults."""

    return StreamPETRForwardConfig(name="streampetr_r50_forward")


def debug_forward_config(
    *,
    num_queries: int = 96,
    decoder_layers: int = 2,
    depth_num: int = 6,
    memory_len: int = 64,
    topk_proposals: int = 16,
    num_propagated: int = 12,
) -> StreamPETRForwardConfig:
    """Small config for fast local smoke tests."""

    base = streampetr_r50_forward_config()
    return replace(
        base,
        name=f"{base.name}_debug",
        num_queries=int(num_queries),
        num_decoder_layers=int(decoder_layers),
        depth_num=int(depth_num),
        max_num=min(100, int(num_queries)),
        memory_len=int(memory_len),
        topk_proposals=min(int(topk_proposals), int(memory_len), int(num_queries)),
        num_propagated=min(int(num_propagated), int(memory_len)),
    )


def replace_backbone(cfg: StreamPETRForwardConfig, **kwargs: object) -> StreamPETRForwardConfig:
    """Return cfg with a replaced BackboneNeckConfig."""

    return replace(cfg, backbone_neck=replace(cfg.backbone_neck, **kwargs))


def replace_config(cfg: StreamPETRForwardConfig, **kwargs: object) -> StreamPETRForwardConfig:
    """Return cfg with selected values replaced."""

    return replace(cfg, **kwargs)

