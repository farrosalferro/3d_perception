"""NMS-free decoder used by BEVFormer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ....common.postprocess.nms_free import NMSFreeDecodeProfile, decode_nms_free_single
from ..utils.boxes import denormalize_bbox

_BEVFORMER_DECODE_PROFILE = NMSFreeDecodeProfile(
    cap_topk_by_numel=False,
    score_threshold_inclusive=False,
    apply_threshold_mask_when_score_truthy=True,
    relax_empty_threshold=True,
    relax_min_threshold=0.01,
)


@dataclass
class NMSFreeCoderLite:
    """A minimal, framework-free variant of BEVFormer NMSFreeCoder."""

    pc_range: Sequence[float]
    post_center_range: Sequence[float] | None
    max_num: int = 300
    score_threshold: float | None = None
    num_classes: int = 10

    def decode_single(self, cls_scores: torch.Tensor, bbox_preds: torch.Tensor) -> dict[str, torch.Tensor]:
        return decode_nms_free_single(
            cls_scores,
            bbox_preds,
            num_classes=self.num_classes,
            max_num=self.max_num,
            score_threshold=self.score_threshold,
            post_center_range=self.post_center_range,
            denormalize_bbox_fn=denormalize_bbox,
            profile=_BEVFORMER_DECODE_PROFILE,
        )

    def decode(self, preds_dicts: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        all_cls_scores = preds_dicts["all_cls_scores"][-1]
        all_bbox_preds = preds_dicts["all_bbox_preds"][-1]
        batch_size = all_cls_scores.size(0)
        return [self.decode_single(all_cls_scores[i], all_bbox_preds[i]) for i in range(batch_size)]
