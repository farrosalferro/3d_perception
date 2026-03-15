"""Standalone StreamPETR forward-only model."""

from __future__ import annotations

import torch
from torch import nn

from .backbone_neck import BackboneNeck
from .config import StreamPETRForwardConfig
from .head import StreamPETRHeadLite
from .transformer import StreamPETRTemporalDecoderLite, StreamPETRTemporalTransformerLite
from .utils import validate_streampetr_img_metas


class StreamPETRLite(nn.Module):
    """Pure-PyTorch StreamPETR implementation for forward-path study/testing."""

    def __init__(self, cfg: StreamPETRForwardConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone_neck = BackboneNeck(cfg.backbone_neck)
        decoder = StreamPETRTemporalDecoderLite(
            num_layers=cfg.num_decoder_layers,
            embed_dims=cfg.embed_dims,
            ffn_dims=cfg.ffn_dims,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            return_intermediate=True,
        )
        transformer = StreamPETRTemporalTransformerLite(decoder=decoder)
        self.head = StreamPETRHeadLite(cfg, transformer)
        self._prev_scene_tokens: list[object] | None = None

    def extract_img_feat(self, img: torch.Tensor, img_metas: list[dict]) -> list[torch.Tensor]:
        """Extract multiscale camera features from [B, Ncam, 3, H, W]."""

        if img.dim() != 5:
            raise ValueError(f"Expected image shape [B, Ncam, 3, H, W], got {tuple(img.shape)}")
        batch_size, num_cams, channels, height, width = img.shape
        validate_streampetr_img_metas(img_metas, batch_size=batch_size, num_cams=num_cams)
        for meta in img_metas:
            meta["input_shape"] = (height, width)

        img_flat = img.reshape(batch_size * num_cams, channels, height, width)
        feats = self.backbone_neck(img_flat)
        return [feat.view(batch_size, num_cams, feat.shape[1], feat.shape[2], feat.shape[3]) for feat in feats]

    def reset_memory(self) -> None:
        self.head.reset_memory()
        self._prev_scene_tokens = None

    def _compute_prev_exists_from_scene_tokens(
        self,
        img_metas: list[dict],
        *,
        batch_size: int,
        num_cams: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        validate_streampetr_img_metas(
            img_metas,
            batch_size=batch_size,
            num_cams=num_cams,
            require_scene_token=True,
        )
        scene_tokens = [meta["scene_token"] for meta in img_metas]
        if self._prev_scene_tokens is None or len(self._prev_scene_tokens) != batch_size:
            prev_exists = torch.zeros(batch_size, device=device, dtype=dtype)
        else:
            prev_exists = torch.tensor(
                [
                    1.0 if scene_tokens[idx] == self._prev_scene_tokens[idx] else 0.0
                    for idx in range(batch_size)
                ],
                device=device,
                dtype=dtype,
            )
        self._prev_scene_tokens = scene_tokens
        return prev_exists

    def forward(
        self,
        img: torch.Tensor,
        img_metas: list[dict],
        *,
        decode: bool = False,
        prev_exists: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | dict[str, object]:
        if not isinstance(img_metas, list):
            raise TypeError(f"img_metas must be a list, got {type(img_metas)}.")
        if img.dim() != 5:
            raise ValueError(f"Expected image shape [B, Ncam, 3, H, W], got {tuple(img.shape)}")
        batch_size, num_cams = img.shape[:2]
        validate_streampetr_img_metas(img_metas, batch_size=batch_size, num_cams=num_cams)

        if prev_exists is None:
            prev_exists = self._compute_prev_exists_from_scene_tokens(
                img_metas,
                batch_size=batch_size,
                num_cams=num_cams,
                device=img.device,
                dtype=img.dtype,
            )
        else:
            if not torch.is_tensor(prev_exists):
                prev_exists = torch.as_tensor(prev_exists, dtype=img.dtype, device=img.device)
            prev_exists = prev_exists.to(device=img.device, dtype=img.dtype).view(-1)
            if prev_exists.numel() != batch_size:
                raise ValueError(
                    f"prev_exists must contain {batch_size} elements, got shape {tuple(prev_exists.shape)}."
                )
            if all("scene_token" in meta for meta in img_metas):
                self._prev_scene_tokens = [meta["scene_token"] for meta in img_metas]

        img_feats = self.extract_img_feat(img, img_metas)
        outputs = self.head(img_feats, img_metas, prev_exists=prev_exists)
        if decode:
            return {"preds": outputs, "decoded": self.head.get_bboxes(outputs)}
        return outputs

    def simple_test(self, img: torch.Tensor, img_metas: list[dict]) -> list[dict[str, torch.Tensor]]:
        outputs = self.forward(img, img_metas, decode=False)
        if not isinstance(outputs, dict):
            raise TypeError("Expected forward outputs to be a dict.")
        return self.head.get_bboxes(outputs)

