"""Standalone BEVerse-style prediction model (forward-only)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .backbone_neck import BackboneNeck
from .config import BEVerseForwardConfig
from .head import TrajectoryHeadLite
from .temporal import TemporalPredictorLite


class BEVerseLite(nn.Module):
    """Pure-PyTorch BEVerse-style trajectory predictor for study/testing."""

    def __init__(self, cfg: BEVerseForwardConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone_neck = BackboneNeck(cfg.backbone_neck)
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(cfg.embed_dims, cfg.embed_dims, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.embed_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.embed_dims, cfg.embed_dims, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.temporal_predictor = TemporalPredictorLite(cfg)
        self.trajectory_head = TrajectoryHeadLite(cfg)

    def extract_img_feat(self, img: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-level camera features from [B, Ncam, 3, H, W]."""

        if img.dim() != 5:
            raise ValueError(f"Expected image shape [B, Ncam, 3, H, W], got {tuple(img.shape)}")
        batch_size, num_cams, channels, height, width = img.shape
        img_flat = img.reshape(batch_size * num_cams, channels, height, width)
        feats = self.backbone_neck(img_flat)
        return [feat.view(batch_size, num_cams, feat.shape[1], feat.shape[2], feat.shape[3]) for feat in feats]

    def _decode_prediction(self, outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mode_probs = outputs["mode_probs"]
        mode_idx = mode_probs.argmax(dim=-1)
        batch_size, _, horizon, _ = outputs["trajectory_preds"].shape
        gather_index = mode_idx.view(batch_size, 1, 1, 1).expand(-1, 1, horizon, 2)
        best_traj = outputs["trajectory_preds"].gather(dim=1, index=gather_index).squeeze(1)
        best_prob = mode_probs.gather(dim=1, index=mode_idx.unsqueeze(-1)).squeeze(-1)
        return {
            "best_mode_idx": mode_idx,
            "best_mode_prob": best_prob,
            "best_trajectory": best_traj,
        }

    def forward(
        self,
        img: torch.Tensor,
        img_metas: list[dict[str, Any]] | None = None,
        *,
        decode: bool = False,
    ) -> dict[str, torch.Tensor] | dict[str, object]:
        del img_metas  # kept for API parity with other models
        img_feats = self.extract_img_feat(img)
        camera_feat = img_feats[0]
        fused_camera_feat = camera_feat.mean(dim=1)
        bev_seed = nn.functional.adaptive_avg_pool2d(
            fused_camera_feat,
            output_size=(self.cfg.bev_h, self.cfg.bev_w),
        )
        bev_embed = self.bev_encoder(bev_seed)
        temporal_tokens = self.temporal_predictor(bev_embed)
        outputs = self.trajectory_head(temporal_tokens)
        outputs.update(
            {
                "camera_feat": camera_feat,
                "bev_embed": bev_embed,
                "temporal_tokens": temporal_tokens,
                "time_stamps": torch.arange(
                    1, self.cfg.pred_horizon + 1, device=img.device, dtype=img.dtype
                )
                * self.cfg.future_dt,
            }
        )
        if decode:
            return {"preds": outputs, "decoded": self._decode_prediction(outputs)}
        return outputs

    def simple_test(
        self,
        img: torch.Tensor,
        img_metas: list[dict[str, Any]] | None = None,
    ) -> dict[str, torch.Tensor]:
        out = self.forward(img, img_metas, decode=True)
        if not isinstance(out, dict) or "decoded" not in out:
            raise TypeError("Expected decoded output dict from forward.")
        decoded = out["decoded"]
        if not isinstance(decoded, dict):
            raise TypeError("Decoded output must be a dictionary.")
        return decoded
