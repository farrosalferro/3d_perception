"""NMS-free decoder used by BEVFormer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ..utils.boxes import denormalize_bbox


@dataclass
class NMSFreeCoderLite:
    """A minimal, framework-free variant of BEVFormer NMSFreeCoder."""

    pc_range: Sequence[float]
    post_center_range: Sequence[float] | None
    max_num: int = 300
    score_threshold: float | None = None
    num_classes: int = 10

    def decode_single(self, cls_scores: torch.Tensor, bbox_preds: torch.Tensor) -> dict[str, torch.Tensor]:
        max_num = self.max_num
        cls_scores = cls_scores.sigmoid()
        scores, indices = cls_scores.view(-1).topk(max_num)
        labels = indices % self.num_classes
        bbox_indices = indices // self.num_classes
        bbox_preds = bbox_preds[bbox_indices]

        final_box_preds = denormalize_bbox(bbox_preds)
        final_scores = scores
        final_preds = labels

        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

        if self.post_center_range is not None:
            post_center_range = torch.as_tensor(self.post_center_range, device=scores.device, dtype=final_box_preds.dtype)
            mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(dim=1)
            mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(dim=1)
            # Keep parity with upstream implementation: only apply threshold
            # mask when score_threshold is truthy (e.g. non-zero float).
            if self.score_threshold:
                mask &= thresh_mask
            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            return {
                "bboxes": boxes3d,
                "scores": scores,
                "labels": labels,
            }
        raise NotImplementedError(
            "Need post_center_range for batched output reorganization; "
            "this pure-PyTorch parity path matches upstream behavior."
        )

    def decode(self, preds_dicts: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        all_cls_scores = preds_dicts["all_cls_scores"][-1]
        all_bbox_preds = preds_dicts["all_bbox_preds"][-1]
        batch_size = all_cls_scores.size(0)
        return [self.decode_single(all_cls_scores[i], all_bbox_preds[i]) for i in range(batch_size)]
