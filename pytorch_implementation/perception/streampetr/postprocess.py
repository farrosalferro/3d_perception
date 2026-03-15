"""NMS-free decode helper for StreamPETR outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .utils import denormalize_bbox


@dataclass
class NMSFreeCoderLite:
    """Minimal StreamPETR-style top-k decode over query/class logits."""

    pc_range: Sequence[float]
    post_center_range: Sequence[float] | None
    max_num: int
    num_classes: int
    score_threshold: float | None = None

    def decode_single(self, cls_scores: torch.Tensor, bbox_preds: torch.Tensor) -> dict[str, torch.Tensor]:
        cls_scores = cls_scores.sigmoid()
        topk = min(self.max_num, cls_scores.numel())
        scores, indices = cls_scores.reshape(-1).topk(topk)
        labels = indices % self.num_classes
        bbox_indices = torch.div(indices, self.num_classes, rounding_mode="floor")
        bbox_preds = bbox_preds[bbox_indices]

        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)
        final_scores = scores
        final_preds = labels

        keep = torch.ones_like(final_scores, dtype=torch.bool)
        if self.score_threshold is not None:
            keep = keep & (final_scores >= self.score_threshold)

        if self.post_center_range is None:
            raise NotImplementedError(
                "Need post_center_range to reorganize output as a batch in NMSFreeCoderLite."
            )

        post_range = torch.as_tensor(
            self.post_center_range,
            dtype=final_box_preds.dtype,
            device=final_box_preds.device,
        )
        center = final_box_preds[..., :3]
        keep = keep & (center >= post_range[:3]).all(dim=1) & (center <= post_range[3:]).all(dim=1)

        return {"bboxes": final_box_preds[keep], "scores": final_scores[keep], "labels": final_preds[keep]}

    def decode(self, preds_dicts: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        all_cls_scores = preds_dicts["all_cls_scores"][-1]
        all_bbox_preds = preds_dicts["all_bbox_preds"][-1]
        batch_size = all_cls_scores.shape[0]
        return [self.decode_single(all_cls_scores[i], all_bbox_preds[i]) for i in range(batch_size)]

