"""Configuration for the pure PyTorch FlashOcc-style predictor."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class FlashOccBackboneConfig:
    """Compact BEV encoder settings for occupancy sequences."""

    in_channels: int = 4
    embed_dims: int = 96
    stem_kernel: int = 3
    stem_stride: int = 2
    stem_padding: int = 1
    num_res_blocks: int = 2


@dataclass(frozen=True)
class FlashOccDepthViewConfig:
    """Depth-aware BEV lifting settings inspired by LSS/BEVDepth."""

    depth_bins: int = 8
    depth_min: float = 1.0
    depth_max: float = 45.0
    context_channels: int = 96
    collapse_z: bool = True


@dataclass(frozen=True)
class FlashOccTemporalWarpConfig:
    """Temporal BEV warp/alignment settings aligned with BEVDet4D semantics."""

    enabled: bool = True
    align_after_view_transformation: bool = True
    x_bounds: tuple[float, float] = (-51.2, 51.2)
    y_bounds: tuple[float, float] = (-51.2, 51.2)


@dataclass(frozen=True)
class FlashOccOccupancyHeadConfig:
    """Occupancy logits head settings mirroring BEVOCCHead2D contracts."""

    dz: int = 8
    num_classes: int = 18
    use_predicter: bool = True
    out_dim: int = 96
    decode_use_gpu: bool = True


@dataclass(frozen=True)
class FlashOccConfig:
    """Forward-only FlashOcc-style trajectory prediction configuration."""

    name: str
    num_history: int = 4
    pred_horizon: int = 12
    dt: float = 0.5
    bev_h: int = 64
    bev_w: int = 80
    num_queries: int = 32
    num_modes: int = 3
    num_heads: int = 4
    topk: int = 8
    backbone: FlashOccBackboneConfig = FlashOccBackboneConfig()
    depth_view: FlashOccDepthViewConfig = FlashOccDepthViewConfig()
    temporal_warp: FlashOccTemporalWarpConfig = FlashOccTemporalWarpConfig()
    occupancy_head: FlashOccOccupancyHeadConfig = FlashOccOccupancyHeadConfig()


def flashocc_base_config() -> FlashOccConfig:
    """Default FlashOcc-style prediction config."""

    return FlashOccConfig(name="flashocc_prediction_base")


def debug_forward_config(
    *,
    num_history: int = 4,
    pred_horizon: int = 8,
    bev_h: int = 48,
    bev_w: int = 64,
    num_queries: int = 12,
    num_modes: int = 3,
    topk: int = 6,
) -> FlashOccConfig:
    """Small config for fast local shape/finite tests."""

    base = flashocc_base_config()
    return replace(
        base,
        name=f"{base.name}_debug",
        num_history=int(num_history),
        pred_horizon=int(pred_horizon),
        bev_h=int(bev_h),
        bev_w=int(bev_w),
        num_queries=int(num_queries),
        num_modes=int(num_modes),
        topk=int(topk),
    )


def replace_backbone(cfg: FlashOccConfig, **kwargs: object) -> FlashOccConfig:
    """Return cfg with replaced backbone values."""

    return replace(cfg, backbone=replace(cfg.backbone, **kwargs))


def replace_config(cfg: FlashOccConfig, **kwargs: object) -> FlashOccConfig:
    """Return cfg with selected fields replaced."""

    return replace(cfg, **kwargs)

