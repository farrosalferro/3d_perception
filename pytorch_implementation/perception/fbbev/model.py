"""Standalone FB-BEV forward-only model."""

from __future__ import annotations

import torch
from torch import nn

from .backbone_neck import BackboneNeck
from .backward_projection import BackwardProjectionLite
from .bev_encoder import BEVEncoderLite
from .config import FBBEVForwardConfig
from .depth_net import FBBEVDepthNetLite
from .detection_head import FBBEVDetectionHeadLite
from .forward_projection import ForwardProjectionLite
from .temporal_fusion import TemporalFusionLite


class FBBEVLite(nn.Module):
    """Pure-PyTorch FB-BEV implementation for forward-path study/testing."""

    def __init__(self, cfg: FBBEVForwardConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone_neck = BackboneNeck(cfg.backbone_neck)
        self.depth_net = FBBEVDepthNetLite(cfg)
        self.forward_projection = ForwardProjectionLite(cfg)
        self.backward_projection = BackwardProjectionLite(cfg)
        self.temporal_fusion = TemporalFusionLite(cfg)
        self.bev_encoder = BEVEncoderLite(cfg)
        self.detection_head = FBBEVDetectionHeadLite(cfg)

    def extract_img_feat(self, img: torch.Tensor) -> list[torch.Tensor]:
        """Extract multiscale camera features from [B, Ncam, 3, H, W]."""

        if img.dim() != 5:
            raise ValueError(f"Expected image shape [B, Ncam, 3, H, W], got {tuple(img.shape)}")
        batch_size, num_cams, channels, height, width = img.shape
        img_flat = img.reshape(batch_size * num_cams, channels, height, width)
        feats = self.backbone_neck(img_flat)
        return [feat.view(batch_size, num_cams, feat.shape[1], feat.shape[2], feat.shape[3]) for feat in feats]

    def _fuse_history(self, bev_refined: torch.Tensor, img_metas: list[dict]) -> torch.Tensor:
        if not self.cfg.use_temporal_fusion:
            return bev_refined
        return self.temporal_fusion(bev_refined, img_metas)

    def forward(
        self,
        img: torch.Tensor,
        img_metas: list[dict],
        *,
        decode: bool = False,
    ) -> dict[str, torch.Tensor] | dict[str, object]:
        img_feats = self.extract_img_feat(img)
        camera_feat = img_feats[0]
        context, depth = self.depth_net(camera_feat)
        bev_volume = self.forward_projection(context, depth)
        bev_refined = self.backward_projection(bev_volume, context, depth)
        bev_fused = self._fuse_history(bev_refined, img_metas)
        bev_embed = self.bev_encoder(bev_fused)
        outputs = self.detection_head(bev_embed)
        outputs.update(
            {
                "context": context,
                "depth": depth,
                "bev_volume": bev_volume,
                "bev_refined": bev_refined,
                "bev_fused": bev_fused,
                "bev_embed": bev_embed,
            }
        )
        if decode:
            return {"preds": outputs, "decoded": self.detection_head.get_bboxes(outputs)}
        return outputs

    def simple_test(self, img: torch.Tensor, img_metas: list[dict]) -> list[dict[str, torch.Tensor]]:
        outputs = self.forward(img, img_metas, decode=False)
        if not isinstance(outputs, dict):
            raise TypeError("Expected forward outputs to be a dict.")
        return self.detection_head.get_bboxes(outputs)

    def clear_temporal_state(self) -> None:
        self.temporal_fusion.clear()
