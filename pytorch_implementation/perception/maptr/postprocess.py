"""NMS-free decode helper for MapTR outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .utils import denormalize_boxes_cxcywh, denormalize_points


@dataclass
class MapTRNMSFreeCoderLite:
    """MapTR-style top-k decode over vector/class logits."""

    post_center_range: Sequence[float] | None
    pc_range: Sequence[float]
    max_num: int
    num_classes: int
    score_threshold: float | None = None

    def decode_single(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        pts_preds: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cls_scores = cls_scores.sigmoid()
        topk = min(self.max_num, cls_scores.numel())
        scores, indices = cls_scores.reshape(-1).topk(topk)
        labels = indices % self.num_classes
        vec_indices = torch.div(indices, self.num_classes, rounding_mode="floor")
        bboxes = bbox_preds[vec_indices]
        pts = pts_preds[vec_indices]

        bboxes_metric = denormalize_boxes_cxcywh(bboxes, self.pc_range)
        pts_metric = denormalize_points(pts, self.pc_range)

        keep = torch.ones_like(scores, dtype=torch.bool)
        thresh_mask: torch.Tensor | None = None
        if self.score_threshold is not None:
            thresh_mask = scores > self.score_threshold
            tmp_score = float(self.score_threshold)
            while int(thresh_mask.sum().item()) == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = scores > -1.0
                    break
                thresh_mask = scores >= tmp_score
            keep &= thresh_mask

        if self.post_center_range is None:
            raise NotImplementedError("post_center_range must be provided for MapTR decode.")

        post = torch.as_tensor(self.post_center_range, dtype=bboxes_metric.dtype, device=bboxes_metric.device)
        if post.numel() == 8:
            keep &= (bboxes_metric[..., :4] >= post[:4]).all(dim=1)
            keep &= (bboxes_metric[..., :4] <= post[4:]).all(dim=1)
        elif post.numel() == 4:
            center_x = 0.5 * (bboxes_metric[:, 0] + bboxes_metric[:, 2])
            center_y = 0.5 * (bboxes_metric[:, 1] + bboxes_metric[:, 3])
            keep &= (center_x >= post[0]) & (center_y >= post[1]) & (center_x <= post[2]) & (center_y <= post[3])
        else:
            raise ValueError(
                "post_center_range must have 4 (center filter) or 8 (xyxy filter) values, "
                f"got {post.numel()}"
            )

        return {
            "bboxes": bboxes_metric[keep],
            "pts": pts_metric[keep],
            "scores": scores[keep],
            "labels": labels[keep],
        }

    def decode(self, preds_dicts: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        all_cls_scores = preds_dicts["all_cls_scores"][-1]
        all_bbox_preds = preds_dicts["all_bbox_preds"][-1]
        all_pts_preds = preds_dicts["all_pts_preds"][-1]
        batch_size = all_cls_scores.shape[0]
        return [
            self.decode_single(all_cls_scores[i], all_bbox_preds[i], all_pts_preds[i])
            for i in range(batch_size)
        ]
