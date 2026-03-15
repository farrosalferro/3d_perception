"""Occupancy-to-BEV encoder for FlashOcc-style prediction."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import FlashOccConfig


class ResidualConvBlock(nn.Module):
    """Small residual block used in BEV feature extraction."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + identity
        return self.act(x)


class FlashOccDepthViewTransformer(nn.Module):
    """Depth-aware BEV lifting with BEVDepth-style depth/context decoupling."""

    def __init__(self, cfg: FlashOccConfig) -> None:
        super().__init__()
        embed_dims = cfg.backbone.embed_dims
        dcfg = cfg.depth_view
        context_channels = dcfg.context_channels if dcfg.context_channels > 0 else embed_dims
        self.depth_bins = int(dcfg.depth_bins)
        self.collapse_z = bool(dcfg.collapse_z)

        self.depth_net = nn.Conv2d(embed_dims, self.depth_bins + context_channels, kernel_size=1, bias=True)
        self.context_proj = nn.Conv2d(context_channels, embed_dims, kernel_size=1, bias=False)
        self.context_norm = nn.BatchNorm2d(embed_dims)
        depth_values = torch.linspace(float(dcfg.depth_min), float(dcfg.depth_max), self.depth_bins)
        self.register_buffer("depth_values", depth_values, persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 4:
            raise ValueError(f"Expected feature map [N, C, H, W], got {tuple(x.shape)}")

        depth_context = self.depth_net(x)
        depth_logits = depth_context[:, : self.depth_bins, ...]
        tran_feat = depth_context[:, self.depth_bins :, ...]
        tran_feat = self.context_proj(tran_feat)
        tran_feat = self.context_norm(tran_feat)
        tran_feat = F.gelu(tran_feat)
        depth_prob = depth_logits.softmax(dim=1)

        # Surrogate lift-splat: distribute context across depth bins, then collapse.
        volume = depth_prob.unsqueeze(1) * tran_feat.unsqueeze(2)
        depth_weight = (self.depth_values / self.depth_values.mean()).view(1, 1, self.depth_bins, 1, 1)
        volume = volume * depth_weight
        if self.collapse_z:
            bev = volume.sum(dim=2)
        else:
            bev = volume.flatten(1, 2)
        return bev, depth_logits, depth_prob


class FlashOccBEVEncoder(nn.Module):
    """Encode occupancy history [B, T, Cin, H, W] into BEV features."""

    def __init__(self, cfg: FlashOccConfig) -> None:
        super().__init__()
        bcfg = cfg.backbone
        self.stem = nn.Sequential(
            nn.Conv2d(
                bcfg.in_channels,
                bcfg.embed_dims,
                kernel_size=bcfg.stem_kernel,
                stride=bcfg.stem_stride,
                padding=bcfg.stem_padding,
                bias=False,
            ),
            nn.BatchNorm2d(bcfg.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([ResidualConvBlock(bcfg.embed_dims) for _ in range(bcfg.num_res_blocks)])
        self.view_transformer = FlashOccDepthViewTransformer(cfg)

    def forward(
        self,
        occ_seq: torch.Tensor,
        *,
        return_depth: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if occ_seq.dim() != 5:
            raise ValueError(f"Expected occupancy sequence [B, T, Cin, H, W], got {tuple(occ_seq.shape)}")
        batch_size, hist, channels, height, width = occ_seq.shape
        x = occ_seq.reshape(batch_size * hist, channels, height, width)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x, depth_logits, depth_prob = self.view_transformer(x)
        _, embed, bev_h, bev_w = x.shape
        bev_sequence = x.view(batch_size, hist, embed, bev_h, bev_w)
        if not return_depth:
            return bev_sequence
        depth_logits = depth_logits.view(batch_size, hist, self.view_transformer.depth_bins, bev_h, bev_w)
        depth_prob = depth_prob.view(batch_size, hist, self.view_transformer.depth_bins, bev_h, bev_w)
        return bev_sequence, depth_logits, depth_prob

