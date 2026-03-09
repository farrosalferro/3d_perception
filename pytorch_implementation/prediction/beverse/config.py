"""Configuration objects for the pure PyTorch BEVerse forward path."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Tuple


@dataclass(frozen=True)
class BackboneNeckConfig:
    """Compact backbone + FPN settings for BEVerse-lite."""

    stage_channels: Tuple[int, int, int, int] = (32, 64, 96, 128)
    out_indices: Tuple[int, ...] = (3,)
    out_channels: int = 128
    num_outs: int = 1


@dataclass(frozen=True)
class BEVerseForwardConfig:
    """Forward-only BEVerse configuration for trajectory prediction study."""

    name: str
    num_cams: int = 6
    embed_dims: int = 128
    bev_h: int = 20
    bev_w: int = 20
    pred_horizon: int = 12
    num_modes: int = 6
    future_dt: float = 0.5
    max_delta: float = 2.0
    backbone_neck: BackboneNeckConfig = BackboneNeckConfig()


def beverse_base_forward_config() -> BEVerseForwardConfig:
    """Return a baseline BEVerse-style forward config."""

    return BEVerseForwardConfig(name="beverse_base_forward")


def debug_forward_config(
    *,
    num_cams: int = 4,
    embed_dims: int = 96,
    bev_h: int = 12,
    bev_w: int = 12,
    pred_horizon: int = 8,
    num_modes: int = 3,
    future_dt: float = 0.5,
) -> BEVerseForwardConfig:
    """Return a small config for fast local tests."""

    base = beverse_base_forward_config()
    backbone = replace(
        base.backbone_neck,
        stage_channels=(24, 48, int(embed_dims), int(embed_dims)),
        out_channels=int(embed_dims),
    )
    return replace(
        base,
        name=f"{base.name}_debug",
        num_cams=int(num_cams),
        embed_dims=int(embed_dims),
        bev_h=int(bev_h),
        bev_w=int(bev_w),
        pred_horizon=int(pred_horizon),
        num_modes=int(num_modes),
        future_dt=float(future_dt),
        backbone_neck=backbone,
    )


def replace_backbone(cfg: BEVerseForwardConfig, **kwargs: object) -> BEVerseForwardConfig:
    """Return cfg with selected BackboneNeckConfig values replaced."""

    return replace(cfg, backbone_neck=replace(cfg.backbone_neck, **kwargs))


def replace_config(cfg: BEVerseForwardConfig, **kwargs: object) -> BEVerseForwardConfig:
    """Return cfg with selected values replaced."""

    return replace(cfg, **kwargs)
