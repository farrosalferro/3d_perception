"""NMS-free decode helper for PETR outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ...common.postprocess.nms_free import NMSFreeDecodeProfile, decode_nms_free_single
from .utils import denormalize_bbox

_PETR_DECODE_PROFILE = NMSFreeDecodeProfile(
    cap_topk_by_numel=False,
    score_threshold_inclusive=False,
    apply_threshold_mask_when_score_truthy=True,
)


@dataclass
class NMSFreeCoderLite:
    """Minimal PETR-style top-k decode over query/class logits."""

    pc_range: Sequence[float]
    post_center_range: Sequence[float] | None
    max_num: int
    num_classes: int
    score_threshold: float | None = None

    def decode_single(self, cls_scores: torch.Tensor, bbox_preds: torch.Tensor) -> dict[str, torch.Tensor]:
        return decode_nms_free_single(
            cls_scores,
            bbox_preds,
            num_classes=self.num_classes,
            max_num=self.max_num,
            score_threshold=self.score_threshold,
            post_center_range=self.post_center_range,
            denormalize_bbox_fn=lambda boxes: denormalize_bbox(boxes, self.pc_range),
            profile=_PETR_DECODE_PROFILE,
        )

    def decode(self, preds_dicts: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        all_cls_scores = preds_dicts["all_cls_scores"][-1]
        all_bbox_preds = preds_dicts["all_bbox_preds"][-1]
        batch_size = all_cls_scores.shape[0]
        return [self.decode_single(all_cls_scores[i], all_bbox_preds[i]) for i in range(batch_size)]

