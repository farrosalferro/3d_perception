"""Standalone PolarFormer forward-only model."""

from __future__ import annotations

import torch
from torch import nn

from .backbone_neck import BackboneNeck
from .config import PolarFormerForwardConfig
from .head import PolarFormerHeadLite
from .transformer import PolarTransformerDecoderLite, PolarTransformerLite


class PolarFormerLite(nn.Module):
    """Pure-PyTorch PolarFormer implementation for forward-path study/testing."""

    def __init__(self, cfg: PolarFormerForwardConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone_neck = BackboneNeck(cfg.backbone_neck, cfg.polar_neck)
        decoder = PolarTransformerDecoderLite(
            num_layers=cfg.num_decoder_layers,
            embed_dims=cfg.embed_dims,
            ffn_dims=cfg.ffn_dims,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            return_intermediate=True,
        )
        transformer = PolarTransformerLite(
            decoder=decoder,
            num_feature_levels=cfg.polar_neck.num_levels,
            embed_dims=cfg.embed_dims,
        )
        self.head = PolarFormerHeadLite(cfg, transformer)

    def extract_img_feat(self, img: torch.Tensor, img_metas: list[dict]) -> list[torch.Tensor]:
        """Extract multi-level polar BEV features from [B, Ncam, 3, H, W]."""

        if img.dim() != 5:
            raise ValueError(f"Expected image shape [B, Ncam, 3, H, W], got {tuple(img.shape)}")
        _ = img_metas  # reserved for future geometry-aware neck extensions
        batch_size, num_cams, channels, height, width = img.shape
        img_flat = img.reshape(batch_size * num_cams, channels, height, width)
        return self.backbone_neck(img_flat, batch_size=batch_size, num_cams=num_cams)

    def forward(
        self,
        img: torch.Tensor,
        img_metas: list[dict],
        *,
        decode: bool = False,
    ) -> dict[str, torch.Tensor] | dict[str, object]:
        mlvl_feats = self.extract_img_feat(img, img_metas)
        outputs = self.head(mlvl_feats, img_metas)
        if decode:
            return {"preds": outputs, "decoded": self.head.get_bboxes(outputs)}
        return outputs

    def simple_test(self, img: torch.Tensor, img_metas: list[dict]) -> list[dict[str, torch.Tensor]]:
        outputs = self.forward(img, img_metas, decode=False)
        if not isinstance(outputs, dict):
            raise TypeError("Expected forward outputs to be a dict.")
        return self.head.get_bboxes(outputs)

