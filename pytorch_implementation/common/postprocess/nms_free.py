"""Shared NMS-free decode primitives with profile adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch


@dataclass(frozen=True)
class NMSFreeDecodeProfile:
    """Adapter knobs for subtle upstream decode differences."""

    cap_topk_by_numel: bool = True
    score_threshold_inclusive: bool = False
    apply_threshold_mask_when_score_truthy: bool = False
    relax_empty_threshold: bool = False
    relax_min_threshold: float = 0.01


def decode_nms_free_single(
    cls_scores: torch.Tensor,
    bbox_preds: torch.Tensor,
    *,
    num_classes: int,
    max_num: int,
    score_threshold: float | None,
    post_center_range: Sequence[float] | None,
    denormalize_bbox_fn: Callable[[torch.Tensor], torch.Tensor],
    profile: NMSFreeDecodeProfile,
) -> dict[str, torch.Tensor]:
    """Decode one sample using profile-configured top-k and filtering behavior."""

    cls_scores = cls_scores.sigmoid()
    topk = int(max_num)
    if profile.cap_topk_by_numel:
        topk = min(topk, int(cls_scores.numel()))
    scores, indices = cls_scores.reshape(-1).topk(topk)
    labels = indices % int(num_classes)
    bbox_indices = torch.div(indices, int(num_classes), rounding_mode="floor")
    bbox_preds = bbox_preds[bbox_indices]

    final_box_preds = denormalize_bbox_fn(bbox_preds)
    final_scores = scores
    final_preds = labels

    thresh_mask: torch.Tensor | None = None
    if score_threshold is not None:
        if profile.score_threshold_inclusive:
            thresh_mask = final_scores >= score_threshold
        else:
            thresh_mask = final_scores > score_threshold
        if profile.relax_empty_threshold:
            tmp_score = float(score_threshold)
            while int(thresh_mask.sum().item()) == 0:
                tmp_score *= 0.9
                if tmp_score < profile.relax_min_threshold:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

    if post_center_range is None:
        raise NotImplementedError(
            "Need post_center_range to reorganize output as a batch in NMSFreeCoderLite."
        )

    post_range = torch.as_tensor(
        post_center_range,
        dtype=final_box_preds.dtype,
        device=final_box_preds.device,
    )
    keep = (final_box_preds[..., :3] >= post_range[:3]).all(dim=1)
    keep &= (final_box_preds[..., :3] <= post_range[3:]).all(dim=1)

    if score_threshold is not None and thresh_mask is not None:
        should_apply = (
            bool(score_threshold) if profile.apply_threshold_mask_when_score_truthy else True
        )
        if should_apply:
            keep &= thresh_mask

    return {
        "bboxes": final_box_preds[keep],
        "scores": final_scores[keep],
        "labels": final_preds[keep],
    }

