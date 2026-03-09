"""Configuration objects for the pure PyTorch BEVFormer forward path."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence, Tuple


@dataclass(frozen=True)
class BackboneNeckConfig:
    """Simple backbone+FPN configuration used by the standalone model."""

    stage_channels: Tuple[int, int, int, int] = (64, 128, 256, 512)
    out_indices: Tuple[int, ...] = (3,)
    num_outs: int = 1
    out_channels: int = 256


@dataclass(frozen=True)
class BEVFormerForwardConfig:
    """Forward-only model configuration."""

    name: str
    pc_range: Tuple[float, float, float, float, float, float]
    post_center_range: Tuple[float, float, float, float, float, float]
    voxel_size: Tuple[float, float, float]
    bev_h: int
    bev_w: int
    num_classes: int = 10
    num_queries: int = 900
    embed_dims: int = 256
    ffn_dims: int = 512
    num_heads: int = 8
    num_cams: int = 6
    num_feature_levels: int = 1
    num_encoder_layers: int = 3
    num_decoder_layers: int = 6
    num_points_in_pillar: int = 4
    spatial_num_points: int = 8
    spatial_num_levels: int = 1
    temporal_num_points: int = 4
    temporal_num_levels: int = 1
    dropout: float = 0.1
    max_num: int = 300
    score_threshold: float | None = None
    backbone_neck: BackboneNeckConfig = BackboneNeckConfig()
    use_cams_embeds: bool = True
    use_can_bus: bool = True
    use_shift: bool = True
    rotate_prev_bev: bool = True
    can_bus_dims: int = 18

    @property
    def real_h(self) -> float:
        return self.pc_range[4] - self.pc_range[1]

    @property
    def real_w(self) -> float:
        return self.pc_range[3] - self.pc_range[0]


def tiny_forward_config() -> BEVFormerForwardConfig:
    """Config aligned with BEVFormer tiny defaults."""

    return BEVFormerForwardConfig(
        name="bevformer_tiny_forward",
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        post_center_range=(-61.2, -61.2, -10.0, 61.2, 61.2, 10.0),
        voxel_size=(0.2, 0.2, 8.0),
        bev_h=50,
        bev_w=50,
        num_feature_levels=1,
        num_encoder_layers=3,
        num_decoder_layers=6,
        spatial_num_levels=1,
        backbone_neck=BackboneNeckConfig(out_indices=(3,), num_outs=1),
    )


def base_forward_config() -> BEVFormerForwardConfig:
    """Config aligned with BEVFormer base defaults."""

    return BEVFormerForwardConfig(
        name="bevformer_base_forward",
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        post_center_range=(-61.2, -61.2, -10.0, 61.2, 61.2, 10.0),
        voxel_size=(0.2, 0.2, 8.0),
        bev_h=200,
        bev_w=200,
        num_feature_levels=4,
        num_encoder_layers=6,
        num_decoder_layers=6,
        spatial_num_levels=4,
        backbone_neck=BackboneNeckConfig(out_indices=(1, 2, 3), num_outs=4),
    )


def debug_forward_config(
    source: str = "tiny",
    *,
    bev_hw: Sequence[int] = (20, 20),
    num_queries: int = 120,
    encoder_layers: int = 2,
    decoder_layers: int = 2,
) -> BEVFormerForwardConfig:
    """Small config for fast local smoke tests."""

    base = tiny_forward_config() if source == "tiny" else base_forward_config()
    if len(bev_hw) != 2:
        raise ValueError("bev_hw must contain exactly two integers.")
    return replace(
        base,
        name=f"{base.name}_debug",
        bev_h=int(bev_hw[0]),
        bev_w=int(bev_hw[1]),
        num_queries=int(num_queries),
        num_encoder_layers=int(encoder_layers),
        num_decoder_layers=int(decoder_layers),
    )
