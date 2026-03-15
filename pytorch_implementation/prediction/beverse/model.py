"""Standalone BEVerse-style prediction model (forward-only)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .backbone_neck import BackboneNeck
from .config import BEVerseForwardConfig
from .head import MultiTaskHeadLite, TrajectoryHeadLite
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
        self.temporal_model = self.temporal_predictor  # naming parity with upstream detector
        self.pts_bbox_head = MultiTaskHeadLite(cfg)
        if not cfg.task_enable_motion or "motion" not in self.pts_bbox_head.task_decoders:
            raise ValueError("BEVerseLite requires motion task enabled for trajectory prediction.")
        self.trajectory_head = self.pts_bbox_head.task_decoders["motion"]
        if not isinstance(self.trajectory_head, TrajectoryHeadLite):
            raise TypeError("Motion decoder must be a TrajectoryHeadLite instance.")

    def _normalize_img(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() == 5:
            img = img.unsqueeze(1)
        if img.dim() != 6:
            raise ValueError(
                "Expected image shape [B, Ncam, 3, H, W] or [B, S, Ncam, 3, H, W], "
                f"got {tuple(img.shape)}"
            )
        if img.shape[3] != 3:
            raise ValueError(f"Expected RGB channels at dim=3, got {img.shape[3]}")
        if self.cfg.strict_meta_validation and img.shape[2] != self.cfg.num_cams:
            raise ValueError(f"Expected num_cams={self.cfg.num_cams}, got {img.shape[2]}")
        return img

    def _validate_time_indices(self, meta: dict[str, Any], seq_len: int) -> None:
        for key in ("frame_indices", "timestamp_indices", "time_indices"):
            if key not in meta:
                continue
            values = meta[key]
            if not isinstance(values, (list, tuple)):
                raise TypeError(f"img_metas[*]['{key}'] must be list/tuple, got {type(values)}")
            if len(values) < seq_len:
                raise ValueError(
                    f"img_metas[*]['{key}'] must contain at least {seq_len} entries."
                )
            index_tensor = torch.as_tensor(values[:seq_len], dtype=torch.float32)
            if index_tensor.numel() > 1 and not bool(torch.all(index_tensor[1:] > index_tensor[:-1])):
                raise ValueError(f"img_metas[*]['{key}'] must be strictly increasing.")

    def _validate_img_metas(
        self,
        img_metas: list[dict[str, Any]] | None,
        batch_size: int,
        seq_len: int,
    ) -> None:
        if img_metas is None:
            return
        if not isinstance(img_metas, list):
            raise TypeError(f"img_metas must be a list, got {type(img_metas)}")
        if len(img_metas) != batch_size:
            raise ValueError(f"Expected {batch_size} img_metas entries, got {len(img_metas)}")
        for meta in img_metas:
            if not isinstance(meta, dict):
                raise TypeError(f"Each img_meta must be dict, got {type(meta)}")
            if self.cfg.strict_meta_validation:
                self._validate_time_indices(meta, seq_len)
                if "img_is_valid" in meta:
                    flags = meta["img_is_valid"]
                    if not isinstance(flags, (list, tuple)):
                        raise TypeError(
                            "img_metas[*]['img_is_valid'] must be list/tuple when provided."
                        )
                    if len(flags) < seq_len:
                        raise ValueError(
                            f"img_metas[*]['img_is_valid'] must contain at least {seq_len} entries."
                        )

    def _resolve_img_is_valid(
        self,
        img_is_valid: torch.Tensor | None,
        img_metas: list[dict[str, Any]] | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if img_is_valid is not None:
            return img_is_valid.to(device=device).bool()
        if not img_metas:
            return None
        if not all(isinstance(meta, dict) and "img_is_valid" in meta for meta in img_metas):
            return None
        rows = []
        for meta in img_metas:
            flags = meta["img_is_valid"]
            if not isinstance(flags, (list, tuple)) or len(flags) < seq_len:
                return None
            rows.append(torch.as_tensor(flags[:seq_len], device=device))
        if len(rows) != batch_size:
            return None
        return torch.stack(rows, dim=0).bool()

    def _validate_future_egomotion(
        self,
        future_egomotion: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if future_egomotion is None:
            return None
        if future_egomotion.dim() != 3:
            raise ValueError(
                f"future_egomotion must be [B, S, D], got {tuple(future_egomotion.shape)}"
            )
        if future_egomotion.shape[0] != batch_size:
            raise ValueError(
                f"future_egomotion batch mismatch: expected {batch_size}, got {future_egomotion.shape[0]}"
            )
        if future_egomotion.shape[1] < seq_len:
            raise ValueError("future_egomotion must provide at least one vector per history step.")
        if future_egomotion.shape[2] < 2:
            raise ValueError("future_egomotion must include XY translation dimensions.")
        return future_egomotion[:, :seq_len].to(device=device)

    def _encode_bev_sequence(self, camera_feat_sequence: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, channels, feat_h, feat_w = camera_feat_sequence.shape
        fused_camera_feat = camera_feat_sequence.mean(dim=2)
        bev_seed = nn.functional.adaptive_avg_pool2d(
            fused_camera_feat.reshape(batch_size * seq_len, channels, feat_h, feat_w),
            output_size=(self.cfg.bev_h, self.cfg.bev_w),
        )
        bev_embed = self.bev_encoder(bev_seed)
        bev_embed = bev_embed.view(batch_size, seq_len, self.cfg.embed_dims, self.cfg.bev_h, self.cfg.bev_w)
        return bev_embed

    def extract_img_feat(
        self,
        img: torch.Tensor,
        img_metas: list[dict[str, Any]] | None = None,
        future_egomotion: torch.Tensor | None = None,
        img_is_valid: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Extract multi-level camera features from batched multi-camera sequences."""
        img_seq = self._normalize_img(img)
        batch_size, seq_len, num_cams, channels, height, width = img_seq.shape
        self._validate_img_metas(img_metas, batch_size, seq_len)
        self._validate_future_egomotion(
            future_egomotion, batch_size=batch_size, seq_len=seq_len, device=img_seq.device
        )
        if img_is_valid is not None:
            if img_is_valid.dim() != 2:
                raise ValueError(f"img_is_valid must be [B, S], got {tuple(img_is_valid.shape)}")
            if img_is_valid.shape[0] != batch_size or img_is_valid.shape[1] < seq_len:
                raise ValueError(
                    f"img_is_valid must have shape [B, >=S], got {tuple(img_is_valid.shape)} for B={batch_size}, S={seq_len}"
                )

        img_flat = img_seq.reshape(batch_size * seq_len * num_cams, channels, height, width)
        feats = self.backbone_neck(img_flat)
        return [
            feat.view(batch_size, seq_len, num_cams, feat.shape[1], feat.shape[2], feat.shape[3])
            for feat in feats
        ]

    def _decode_prediction(self, outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.trajectory_head.inference(outputs)

    def _build_time_stamps(
        self,
        device: torch.device,
        dtype: torch.dtype,
        img_metas: list[dict[str, Any]] | None,
    ) -> torch.Tensor:
        if img_metas and "future_time_stamps" in img_metas[0]:
            values = img_metas[0]["future_time_stamps"]
            if isinstance(values, (list, tuple)) and len(values) >= self.cfg.pred_horizon:
                time_stamps = torch.as_tensor(
                    values[: self.cfg.pred_horizon],
                    device=device,
                    dtype=dtype,
                )
            else:
                raise ValueError(
                    "img_metas[0]['future_time_stamps'] must be list/tuple with pred_horizon entries."
                )
        else:
            time_stamps = (
                torch.arange(1, self.cfg.pred_horizon + 1, device=device, dtype=dtype)
                * self.cfg.future_dt
            )
        if self.cfg.strict_meta_validation and time_stamps.numel() > 1:
            if not bool(torch.all(time_stamps[1:] > time_stamps[:-1])):
                raise ValueError("time_stamps must be strictly increasing.")
        return time_stamps

    def forward(
        self,
        img: torch.Tensor,
        img_metas: list[dict[str, Any]] | None = None,
        *,
        future_egomotion: torch.Tensor | None = None,
        img_is_valid: torch.Tensor | None = None,
        decode: bool = False,
        rescale: bool = False,
    ) -> dict[str, torch.Tensor] | dict[str, object]:
        img_feats = self.extract_img_feat(
            img,
            img_metas=img_metas,
            future_egomotion=future_egomotion,
            img_is_valid=img_is_valid,
        )
        camera_feat_sequence = img_feats[0]
        batch_size, seq_len = camera_feat_sequence.shape[:2]

        resolved_img_is_valid = self._resolve_img_is_valid(
            img_is_valid,
            img_metas=img_metas,
            batch_size=batch_size,
            seq_len=seq_len,
            device=img.device,
        )
        future_egomotion = self._validate_future_egomotion(
            future_egomotion, batch_size=batch_size, seq_len=seq_len, device=img.device
        )

        bev_embed_sequence = self._encode_bev_sequence(camera_feat_sequence)
        temporal_tokens = self.temporal_predictor(
            bev_embed_sequence,
            future_egomotion=future_egomotion,
            img_is_valid=resolved_img_is_valid,
        )
        multi_task_predictions = self.pts_bbox_head(
            bev_embed_sequence[:, -1],
            targets=None,
            temporal_tokens=temporal_tokens,
        )
        if "motion" not in multi_task_predictions:
            raise KeyError("Expected motion predictions from pts_bbox_head.")

        outputs: dict[str, Any] = dict(multi_task_predictions["motion"])
        outputs.update(
            {
                "camera_feat": camera_feat_sequence[:, -1],
                "camera_feat_sequence": camera_feat_sequence,
                "bev_embed": bev_embed_sequence[:, -1],
                "bev_embed_sequence": bev_embed_sequence,
                "temporal_tokens": temporal_tokens,
                "time_stamps": self._build_time_stamps(
                    device=img.device,
                    dtype=img.dtype,
                    img_metas=img_metas,
                ),
                "multi_task_predictions": multi_task_predictions,
            }
        )
        if decode:
            decoded = self._decode_prediction(multi_task_predictions["motion"])
            multi_task_decoded = self.pts_bbox_head.inference(
                multi_task_predictions,
                img_metas=img_metas,
                rescale=rescale,
            )
            return {
                "preds": outputs,
                "decoded": decoded,
                "multi_task_decoded": multi_task_decoded,
            }
        return outputs

    def simple_test(
        self,
        img: torch.Tensor,
        img_metas: list[dict[str, Any]] | None = None,
        future_egomotion: torch.Tensor | None = None,
        img_is_valid: torch.Tensor | None = None,
        rescale: bool = False,
    ) -> dict[str, torch.Tensor]:
        out = self.forward(
            img,
            img_metas,
            future_egomotion=future_egomotion,
            img_is_valid=img_is_valid,
            decode=True,
            rescale=rescale,
        )
        if not isinstance(out, dict) or "decoded" not in out:
            raise TypeError("Expected decoded output dict from forward.")
        decoded = out["decoded"]
        if not isinstance(decoded, dict):
            raise TypeError("Decoded output must be a dictionary.")
        return decoded
