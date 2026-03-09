"""Configuration objects for the pure PyTorch Sparse4D forward path."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple


@dataclass(frozen=True)
class BackboneNeckConfig:
    """Compact image encoder config for the standalone Sparse4D model."""

    stage_channels: Tuple[int, int, int, int] = (64, 128, 192, 256)
    out_indices: Tuple[int, ...] = (0, 1, 2, 3)
    out_channels: int = 256
    num_outs: int = 4


@dataclass(frozen=True)
class Sparse4DForwardConfig:
    """Forward-only Sparse4D configuration."""

    name: str
    num_classes: int = 10
    num_queries: int = 300
    num_cams: int = 6
    embed_dims: int = 256
    ffn_dims: int = 512
    num_heads: int = 8
    num_decoder_layers: int = 6
    box_code_size: int = 11
    max_detections: int = 100
    dropout: float = 0.1
    backbone_neck: BackboneNeckConfig = BackboneNeckConfig()


def sparse4d_forward_config() -> Sparse4DForwardConfig:
    """Sparse4D-like defaults for educational forward study."""

    return Sparse4DForwardConfig(name="sparse4d_forward")


def debug_forward_config(
    *,
    num_queries: int = 48,
    decoder_layers: int = 2,
) -> Sparse4DForwardConfig:
    """Small config for fast local tensor-shape tests."""

    base = sparse4d_forward_config()
    return replace(
        base,
        name=f"{base.name}_debug",
        num_queries=int(num_queries),
        num_decoder_layers=int(decoder_layers),
        max_detections=min(64, int(num_queries)),
    )
