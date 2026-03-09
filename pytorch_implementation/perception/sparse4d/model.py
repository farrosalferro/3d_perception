"""Standalone Sparse4D forward-only model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .backbone_neck import BackboneNeck
from .config import Sparse4DForwardConfig
from .head import Sparse4DHeadLite


class Sparse4DLite(nn.Module):
    """Pure-PyTorch Sparse4D implementation for forward-path study/testing."""

    def __init__(self, cfg: Sparse4DForwardConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone_neck = BackboneNeck(cfg.backbone_neck)
        self.head = Sparse4DHeadLite(cfg)

    def extract_img_feat(self, img: torch.Tensor) -> list[torch.Tensor]:
        """Extract multilevel camera features from [B, Ncam, 3, H, W]."""

        if img.dim() != 5:
            raise ValueError(f"Expected image shape [B, Ncam, 3, H, W], got {tuple(img.shape)}")
        batch_size, num_cams, channels, height, width = img.shape
        if num_cams != self.cfg.num_cams:
            raise ValueError(f"Expected {self.cfg.num_cams} cameras, got {num_cams}.")
        img_flat = img.reshape(batch_size * num_cams, channels, height, width)
        feats = self.backbone_neck(img_flat)
        return [
            feat.view(batch_size, num_cams, feat.shape[1], feat.shape[2], feat.shape[3]) for feat in feats
        ]

    def forward(
        self,
        img: torch.Tensor,
        metas: Any = None,
        *,
        decode: bool = False,
    ) -> dict[str, torch.Tensor] | dict[str, object]:
        img_feats = self.extract_img_feat(img)
        outputs = self.head(img_feats, metas)
        if decode:
            return {"preds": outputs, "decoded": self.head.get_bboxes(outputs)}
        return outputs

    def simple_test(self, img: torch.Tensor, metas: Any = None) -> list[dict[str, torch.Tensor]]:
        outputs = self.forward(img, metas, decode=False)
        if not isinstance(outputs, dict):
            raise TypeError("Expected forward outputs to be a dict.")
        return self.head.get_bboxes(outputs)
