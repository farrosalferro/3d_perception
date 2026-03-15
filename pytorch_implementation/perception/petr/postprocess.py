"""NMS-free decode helper for PETR outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .utils import denormalize_bbox


@dataclass
class NMSFreeCoderLite:
    """Minimal PETR-style top-k decode over query/class logits."""

    pc_range: Sequence[float]
    post_center_range: Sequence[float] | None
    max_num: int
    num_classes: int
    score_threshold: float | None = None

    def decode_single(self, cls_scores: torch.Tensor, bbox_preds: torch.Tensor) -> dict[str, torch.Tensor]:
        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.reshape(-1).topk(self.max_num)
        labels = indexs % self.num_classes
        bbox_indices = torch.div(indexs, self.num_classes, rounding_mode="floor")
        bbox_preds = bbox_preds[bbox_indices]

        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)
        final_scores = scores
        final_preds = labels

        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
        if self.post_center_range is not None:
            post_range = torch.as_tensor(
                self.post_center_range,
                dtype=final_box_preds.dtype,
                device=final_box_preds.device,
            )
            mask = (final_box_preds[..., :3] >= post_range[:3]).all(dim=1)
            mask &= (final_box_preds[..., :3] <= post_range[3:]).all(dim=1)
            if self.score_threshold:
                mask &= thresh_mask

            predictions = {
                "bboxes": final_box_preds[mask],
                "scores": final_scores[mask],
                "labels": final_preds[mask],
            }
            return predictions
        raise NotImplementedError(
            "Need post_center_range to reorganize output as a batch in NMSFreeCoderLite."
        )

    def decode(self, preds_dicts: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        all_cls_scores = preds_dicts["all_cls_scores"][-1]
        all_bbox_preds = preds_dicts["all_bbox_preds"][-1]
        batch_size = all_cls_scores.shape[0]
        return [self.decode_single(all_cls_scores[i], all_bbox_preds[i]) for i in range(batch_size)]

