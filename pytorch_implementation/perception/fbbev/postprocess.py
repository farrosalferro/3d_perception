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
        if cls_scores.dim() != 2 or bbox_preds.dim() != 2:
            raise ValueError(
                "decode_single expects cls_scores [Q, Ccls] and bbox_preds [Q, Cbox], "
                f"got {tuple(cls_scores.shape)} and {tuple(bbox_preds.shape)}"
            )
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

    def decode_occupancy_single(
        self,
        occupancy_logits: torch.Tensor,
        *,
        fix_void: bool = False,
        return_raw_occ: bool = False,
    ) -> torch.Tensor:
        """Decode one sample occupancy logits [Cocc, H, W, Z]."""

        if occupancy_logits.dim() != 4:
            raise ValueError(f"Expected occupancy logits [Cocc, H, W, Z], got {tuple(occupancy_logits.shape)}")
        if not torch.isfinite(occupancy_logits).all():
            raise ValueError("occupancy logits must be finite before decode.")

        occ = occupancy_logits.permute(1, 2, 3, 0).contiguous()  # [H, W, Z, C]
        if fix_void and occ.shape[-1] > 1:
            occ = occ[..., 1:]
        occ = occ.softmax(dim=-1)

        # Match FB-BEV submission-axis conversion.
        occ = occ.permute(3, 2, 0, 1)
        occ = torch.flip(occ, dims=[2])
        occ = torch.rot90(occ, k=-1, dims=[2, 3])
        occ = occ.permute(2, 3, 1, 0).contiguous()
        if return_raw_occ:
            return occ
        return occ.argmax(dim=-1)

    def decode_occupancy(
        self,
        occupancy_logits: torch.Tensor,
        *,
        fix_void: bool = False,
        return_raw_occ: bool = False,
    ) -> list[torch.Tensor]:
        """Decode occupancy logits [B, Cocc, H, W, Z] per sample."""

        if occupancy_logits.dim() != 5:
            raise ValueError(f"Expected occupancy logits [B, Cocc, H, W, Z], got {tuple(occupancy_logits.shape)}")
        return [
            self.decode_occupancy_single(sample, fix_void=fix_void, return_raw_occ=return_raw_occ)
            for sample in occupancy_logits
        ]
