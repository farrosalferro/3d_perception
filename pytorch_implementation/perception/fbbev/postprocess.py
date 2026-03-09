"""NMS-free decode helper for FB-BEV outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class FBBEVBoxCoderLite:
    """Minimal top-k decode over class logits and bbox predictions."""

    post_center_range: Sequence[float] | None
    max_num: int
    num_classes: int
    score_threshold: float | None = None

    def decode_single(self, cls_scores: torch.Tensor, bbox_preds: torch.Tensor) -> dict[str, torch.Tensor]:
        probs = cls_scores.sigmoid()
        flat = probs.reshape(-1)
        topk = min(self.max_num, int(flat.shape[0]))
        scores, indices = flat.topk(topk)
        labels = indices % self.num_classes
        bbox_indices = torch.div(indices, self.num_classes, rounding_mode="floor")
        bboxes = bbox_preds[bbox_indices]

        keep = torch.ones_like(scores, dtype=torch.bool)
        if self.score_threshold is not None:
            keep = keep & (scores > self.score_threshold)
        if self.post_center_range is not None:
            post_range = torch.as_tensor(self.post_center_range, dtype=bboxes.dtype, device=bboxes.device)
            center = bboxes[..., :3]
            keep = keep & (center >= post_range[:3]).all(dim=1) & (center <= post_range[3:]).all(dim=1)
        return {"bboxes": bboxes[keep], "scores": scores[keep], "labels": labels[keep]}

    def decode(self, preds_dicts: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        all_cls_scores = preds_dicts["all_cls_scores"][-1]
        all_bbox_preds = preds_dicts["all_bbox_preds"][-1]
        batch_size = all_cls_scores.shape[0]
        return [self.decode_single(all_cls_scores[i], all_bbox_preds[i]) for i in range(batch_size)]
