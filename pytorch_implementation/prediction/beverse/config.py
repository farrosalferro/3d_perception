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
    temporal_receptive_field: int = 3
    strict_meta_validation: bool = True
    decode_topk: int = 3
    task_enable_map: bool = False
    task_enable_3dod: bool = False
    task_enable_motion: bool = True
    task_weight_map: float = 1.0
    task_weight_3dod: float = 1.0
    task_weight_motion: float = 1.0
    backbone_neck: BackboneNeckConfig = BackboneNeckConfig()

    def __post_init__(self) -> None:
        if self.num_cams <= 0:
            raise ValueError("num_cams must be positive.")
        if self.embed_dims <= 0:
            raise ValueError("embed_dims must be positive.")
        if self.bev_h <= 0 or self.bev_w <= 0:
            raise ValueError("bev_h and bev_w must be positive.")
        if self.pred_horizon <= 0:
            raise ValueError("pred_horizon must be positive.")
        if self.num_modes <= 0:
            raise ValueError("num_modes must be positive.")
        if self.future_dt <= 0:
            raise ValueError("future_dt must be positive.")
        if self.max_delta <= 0:
            raise ValueError("max_delta must be positive.")
        if self.temporal_receptive_field <= 0:
            raise ValueError("temporal_receptive_field must be positive.")
        if self.decode_topk <= 0:
            raise ValueError("decode_topk must be positive.")


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
