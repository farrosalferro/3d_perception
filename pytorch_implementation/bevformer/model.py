"""Standalone BEVFormer forward-only model."""

from __future__ import annotations

import torch
from torch import nn

from .backbone_neck import BackboneNeck
from .config import BEVFormerForwardConfig
from .head import BEVFormerHeadLite
from .modules import BEVFormerEncoderLite, DetectionTransformerDecoderLite, PerceptionTransformerLite


class BEVFormerLite(nn.Module):
    """Pure-PyTorch BEVFormer (forward path only)."""

    def __init__(self, cfg: BEVFormerForwardConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone_neck = BackboneNeck(cfg.backbone_neck)

        encoder = BEVFormerEncoderLite(
            num_layers=cfg.num_encoder_layers,
            pc_range=cfg.pc_range,
            num_points_in_pillar=cfg.num_points_in_pillar,
            return_intermediate=False,
            embed_dims=cfg.embed_dims,
            ffn_dims=cfg.ffn_dims,
            dropout=cfg.dropout,
            num_cams=cfg.num_cams,
            num_heads=cfg.num_heads,
            temporal_num_levels=cfg.temporal_num_levels,
            temporal_num_points=cfg.temporal_num_points,
            spatial_num_levels=cfg.spatial_num_levels,
            spatial_num_points=cfg.spatial_num_points,
        )
        decoder = DetectionTransformerDecoderLite(
            num_layers=cfg.num_decoder_layers,
            embed_dims=cfg.embed_dims,
            ffn_dims=cfg.ffn_dims,
            num_heads=cfg.num_heads,
            return_intermediate=True,
            cross_num_levels=1,
            cross_num_points=4,
            dropout=cfg.dropout,
        )
        transformer = PerceptionTransformerLite(
            encoder=encoder,
            decoder=decoder,
            embed_dims=cfg.embed_dims,
            num_feature_levels=cfg.num_feature_levels,
            num_cams=cfg.num_cams,
            rotate_prev_bev=cfg.rotate_prev_bev,
            use_shift=cfg.use_shift,
            use_can_bus=cfg.use_can_bus,
            use_cams_embeds=cfg.use_cams_embeds,
            can_bus_dims=cfg.can_bus_dims,
        )
        self.head = BEVFormerHeadLite(cfg=cfg, transformer=transformer)

    def extract_img_feat(self, img: torch.Tensor) -> list[torch.Tensor]:
        """Extract multiscale camera features.

        Args:
            img: [B, Ncam, 3, H, W]
        Returns:
            list of [B, Ncam, C, H_l, W_l]
        """

        if img.dim() != 5:
            raise ValueError(f"Expected image shape [B, Ncam, 3, H, W], got {tuple(img.shape)}")
        batch_size, num_cams, channels, height, width = img.shape
        img_flat = img.reshape(batch_size * num_cams, channels, height, width)
        img_feats = self.backbone_neck(img_flat)

        reshaped = []
        for feat in img_feats:
            _, feat_channels, feat_h, feat_w = feat.shape
            reshaped.append(feat.view(batch_size, num_cams, feat_channels, feat_h, feat_w))
        return reshaped

    def forward(
        self,
        img: torch.Tensor,
        img_metas: list[dict],
        *,
        prev_bev: torch.Tensor | None = None,
        only_bev: bool = False,
        decode: bool = False,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        img_feats = self.extract_img_feat(img)
        outputs = self.head(img_feats, img_metas, prev_bev=prev_bev, only_bev=only_bev)
        if only_bev:
            return outputs
        if decode:
            return {
                "preds": outputs,
                "decoded": self.head.get_bboxes(outputs),
            }
        return outputs

    def simple_test(
        self,
        img: torch.Tensor,
        img_metas: list[dict],
        *,
        prev_bev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        outputs = self.forward(img, img_metas, prev_bev=prev_bev, only_bev=False, decode=False)
        if not isinstance(outputs, dict):
            raise TypeError("Expected dict outputs for simple_test.")
        decoded = self.head.get_bboxes(outputs)
        return outputs["bev_embed"], decoded
