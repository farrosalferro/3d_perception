"""Standalone PETR forward-only model."""

from __future__ import annotations

import torch
from torch import nn

from .backbone_neck import BackboneNeck
from .config import PETRForwardConfig
from .head import PETRHeadLite
from .transformer import PETRTransformerDecoderLite, PETRTransformerLite


class PETRLite(nn.Module):
    """Pure-PyTorch PETR implementation for forward-path study/testing."""

    def __init__(self, cfg: PETRForwardConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone_neck = BackboneNeck(cfg.backbone_neck)
        decoder = PETRTransformerDecoderLite(
            num_layers=cfg.num_decoder_layers,
            embed_dims=cfg.embed_dims,
            ffn_dims=cfg.ffn_dims,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            return_intermediate=True,
        )
        transformer = PETRTransformerLite(decoder=decoder)
        self.head = PETRHeadLite(cfg, transformer)

    def extract_img_feat(self, img: torch.Tensor) -> list[torch.Tensor]:
        """Extract multiscale camera features from [B, Ncam, 3, H, W]."""

        if img.dim() != 5:
            raise ValueError(f"Expected image shape [B, Ncam, 3, H, W], got {tuple(img.shape)}")
        batch_size, num_cams, channels, height, width = img.shape
        img_flat = img.reshape(batch_size * num_cams, channels, height, width)
        feats = self.backbone_neck(img_flat)
        return [feat.view(batch_size, num_cams, feat.shape[1], feat.shape[2], feat.shape[3]) for feat in feats]

    def forward(
        self,
        img: torch.Tensor,
        img_metas: list[dict],
        *,
        decode: bool = False,
    ) -> dict[str, torch.Tensor] | dict[str, object]:
        img_feats = self.extract_img_feat(img)
        outputs = self.head(img_feats, img_metas)
        if decode:
            return {"preds": outputs, "decoded": self.head.get_bboxes(outputs)}
        return outputs

    def simple_test(self, img: torch.Tensor, img_metas: list[dict]) -> list[dict[str, torch.Tensor]]:
        outputs = self.forward(img, img_metas, decode=False)
        if not isinstance(outputs, dict):
            raise TypeError("Expected forward outputs to be a dict.")
        return self.head.get_bboxes(outputs)

