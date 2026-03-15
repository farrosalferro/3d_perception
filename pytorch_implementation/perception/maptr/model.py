"""Standalone MapTR forward-only model."""

from __future__ import annotations

import torch
from torch import nn

from .backbone_neck import BackboneNeck
from .config import MapTRForwardConfig
from .head import MapTRHeadLite
from .transformer import MapTRBEVEncoderLite, MapTRDecoderLite, MapTRPerceptionTransformerLite


class MapTRLite(nn.Module):
    """Pure-PyTorch MapTR implementation for forward-path study/testing."""

    def __init__(self, cfg: MapTRForwardConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone_neck = BackboneNeck(cfg.backbone_neck)

        decoder = MapTRDecoderLite(
            num_layers=cfg.num_decoder_layers,
            embed_dims=cfg.embed_dims,
            ffn_dims=cfg.ffn_dims,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            return_intermediate=True,
        )
        bev_encoder = MapTRBEVEncoderLite(
            embed_dims=cfg.embed_dims,
            num_heads=cfg.num_heads,
            ffn_dims=cfg.ffn_dims,
            dropout=cfg.dropout,
        )
        transformer = MapTRPerceptionTransformerLite(
            decoder=decoder,
            bev_encoder=bev_encoder,
            num_cams=cfg.num_cams,
        )
        self.head = MapTRHeadLite(cfg, transformer)

    def _validate_img_metas(self, img_metas: list[dict], *, batch_size: int) -> None:
        if not isinstance(img_metas, list):
            raise TypeError(f"img_metas must be a list[dict], got {type(img_metas)}")
        if len(img_metas) != batch_size:
            raise ValueError(f"Expected {batch_size} img_metas entries, got {len(img_metas)}")
        for batch_idx, meta in enumerate(img_metas):
            if not isinstance(meta, dict):
                raise TypeError(f"img_metas[{batch_idx}] must be dict, got {type(meta)}")
            if self.cfg.strict_img_meta and "img_shape" not in meta:
                raise KeyError(f"img_metas[{batch_idx}] must include 'img_shape'.")
            if self.cfg.strict_img_meta and "pad_shape" not in meta:
                raise KeyError(f"img_metas[{batch_idx}] must include 'pad_shape'.")

    def extract_img_feat(self, img: torch.Tensor) -> list[torch.Tensor]:
        """Extract multiscale camera features from [B, Ncam, 3, H, W]."""

        if img.dim() != 5:
            raise ValueError(f"Expected image shape [B, Ncam, 3, H, W], got {tuple(img.shape)}")
        batch_size, num_cams, channels, height, width = img.shape
        if self.cfg.strict_img_meta and num_cams != self.cfg.num_cams:
            raise ValueError(f"Expected num_cams={self.cfg.num_cams}, got {num_cams}")
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
        self._validate_img_metas(img_metas, batch_size=img.shape[0])
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
