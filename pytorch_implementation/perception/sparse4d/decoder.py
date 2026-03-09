"""Sparse4D decoder and lightweight post-processing."""

from __future__ import annotations

import torch
from torch import nn

from .blocks import SparseBox3DRefinementLite, SparseDecoderLayerLite


class Sparse4DDecoderLite(nn.Module):
    """Layered Sparse4D decoder with iterative box refinement."""

    def __init__(
        self,
        *,
        num_layers: int,
        embed_dims: int,
        ffn_dims: int,
        num_heads: int,
        num_classes: int,
        box_code_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SparseDecoderLayerLite(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    ffn_dims=ffn_dims,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.refinement_layers = nn.ModuleList(
            [SparseBox3DRefinementLite(embed_dims, num_classes, box_code_size) for _ in range(num_layers)]
        )

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor_embed: torch.Tensor,
        anchors: torch.Tensor,
        img_feats: list[torch.Tensor],
        *,
        projection_mat: torch.Tensor | None = None,
        image_wh: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = instance_feature + anchor_embed
        cur_anchor = anchors
        all_cls_scores: list[torch.Tensor] = []
        all_bbox_preds: list[torch.Tensor] = []

        for layer, refine in zip(self.layers, self.refinement_layers):
            query = layer(query, img_feats, projection_mat=projection_mat, image_wh=image_wh)
            cls_scores, bbox_preds = refine(query, cur_anchor)
            all_cls_scores.append(cls_scores)
            all_bbox_preds.append(bbox_preds)
            cur_anchor = bbox_preds.detach()

        return torch.stack(all_cls_scores, dim=0), torch.stack(all_bbox_preds, dim=0)


class SparseBox3DDecoderLite(nn.Module):
    """NMS-free top-k decoder for Sparse4D-style predictions."""

    def __init__(self, max_detections: int = 100) -> None:
        super().__init__()
        self.max_detections = int(max_detections)

    def _to_boxes10(self, raw_box: torch.Tensor) -> torch.Tensor:
        code_size = raw_box.shape[-1]
        x = raw_box[..., 0]
        y = raw_box[..., 1]
        z = raw_box[..., 2]
        w = raw_box[..., 3]
        l = raw_box[..., 4]
        h = raw_box[..., 5]
        if code_size >= 8:
            yaw = torch.atan2(raw_box[..., 6], raw_box[..., 7])
        elif code_size >= 7:
            yaw = raw_box[..., 6]
        else:
            yaw = torch.zeros_like(x)
        vx = raw_box[..., 8] if code_size >= 9 else torch.zeros_like(x)
        vy = raw_box[..., 9] if code_size >= 10 else torch.zeros_like(x)
        vz = raw_box[..., 10] if code_size >= 11 else torch.zeros_like(x)
        return torch.stack([x, y, z, w, l, h, yaw, vx, vy, vz], dim=-1)

    def decode(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        if cls_scores.dim() != 3 or bbox_preds.dim() != 3:
            raise ValueError("Expected cls_scores and bbox_preds with shape [B, Q, C].")
        batch_size, num_queries, num_classes = cls_scores.shape
        k = min(self.max_detections, num_queries * num_classes)

        probs = cls_scores.sigmoid()
        flat = probs.reshape(batch_size, -1)
        topk_scores, topk_indices = torch.topk(flat, k=k, dim=1)
        topk_labels = topk_indices % num_classes
        topk_queries = torch.div(topk_indices, num_classes, rounding_mode="floor")

        outputs: list[dict[str, torch.Tensor]] = []
        for batch_idx in range(batch_size):
            selected = bbox_preds[batch_idx].index_select(0, topk_queries[batch_idx])
            boxes_3d = self._to_boxes10(selected)
            outputs.append(
                {
                    "boxes_3d": boxes_3d,
                    "scores_3d": topk_scores[batch_idx],
                    "labels_3d": topk_labels[batch_idx].to(dtype=torch.long),
                }
            )
        return outputs
