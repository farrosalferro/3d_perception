"""A lightweight image backbone + neck used for FB-BEV forward tests."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .config import BackboneNeckConfig


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleBackbone(nn.Module):
    """Compact CNN backbone with four stages."""

    def __init__(self, stage_channels: Sequence[int]) -> None:
        super().__init__()
        if len(stage_channels) != 4:
            raise ValueError("SimpleBackbone expects exactly four stage channels.")
        c1, c2, c3, c4 = stage_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.stages = nn.ModuleList(
            [
                _ConvBlock(c1, c1, stride=1),
                _ConvBlock(c1, c2, stride=2),
                _ConvBlock(c2, c3, stride=2),
                _ConvBlock(c3, c4, stride=2),
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


class SimpleFPN(nn.Module):
    """Minimal FPN with optional extra levels."""

    def __init__(self, in_channels: Sequence[int], out_channels: int, num_outs: int) -> None:
        super().__init__()
        if num_outs < len(in_channels):
            raise ValueError("num_outs must be >= number of input feature levels.")
        self.lateral_convs = nn.ModuleList([nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels])
        self.output_convs = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels]
        )
        self.extra_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
                for _ in range(num_outs - len(in_channels))
            ]
        )

    def forward(self, inputs: Iterable[torch.Tensor]) -> list[torch.Tensor]:
        feats = list(inputs)
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, feats)]
        for idx in range(len(laterals) - 1, 0, -1):
            laterals[idx - 1] = laterals[idx - 1] + F.interpolate(
                laterals[idx],
                size=laterals[idx - 1].shape[-2:],
                mode="nearest",
            )
        outs = [conv(x) for conv, x in zip(self.output_convs, laterals)]
        if self.extra_convs:
            cur = outs[-1]
            for conv in self.extra_convs:
                cur = conv(cur)
                outs.append(cur)
        return outs


class BackboneNeck(nn.Module):
    """Backbone + neck wrapper that mirrors FB-BEV image feature API."""

    def __init__(self, cfg: BackboneNeckConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = SimpleBackbone(cfg.stage_channels)
        selected_in_channels = [cfg.stage_channels[idx] for idx in cfg.out_indices]
        self.neck = SimpleFPN(
            in_channels=selected_in_channels,
            out_channels=cfg.out_channels,
            num_outs=cfg.num_outs,
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        stages = self.backbone(x)
        selected = [stages[idx] for idx in self.cfg.out_indices]
        return self.neck(selected)
