"""Sparse4D decoder and decode/post-process heads."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from .blocks import SparseBox3DRefinementLite, SparseDecoderLayerLite

CNS = 0
SIN_YAW = 6
COS_YAW = 7
VX = 8


class Sparse4DDecoderLite(nn.Module):
    """Layered Sparse4D decoder with iterative anchor refinement."""

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
        num_levels: int,
        num_cams: int,
        num_single_frame_decoder: int = 1,
        normalize_yaw: bool = False,
        refine_yaw: bool = True,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers)
        self.num_single_frame_decoder = int(num_single_frame_decoder)
        self.layers = nn.ModuleList(
            [
                SparseDecoderLayerLite(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    ffn_dims=ffn_dims,
                    dropout=dropout,
                    box_code_size=box_code_size,
                    num_levels=num_levels,
                    num_cams=num_cams,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.refinement_layers = nn.ModuleList(
            [
                SparseBox3DRefinementLite(
                    embed_dims,
                    num_classes,
                    box_code_size,
                    normalize_yaw=normalize_yaw,
                    refine_yaw=refine_yaw,
                )
                for _ in range(self.num_layers)
            ]
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
        time_interval: torch.Tensor | float = 1.0,
        temp_instance_feature: torch.Tensor | None = None,
        anchor_encoder: nn.Module | None = None,
        instance_bank_update: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ]
        | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query = instance_feature
        cur_anchor = anchors
        cur_anchor_embed = anchor_embed
        all_cls_scores: list[torch.Tensor] = []
        all_bbox_preds: list[torch.Tensor] = []

        for layer_idx, (layer, refine) in enumerate(zip(self.layers, self.refinement_layers)):
            temporal_memory = None
            if temp_instance_feature is not None and layer_idx >= self.num_single_frame_decoder:
                temporal_memory = temp_instance_feature
            query = layer(
                query,
                cur_anchor,
                cur_anchor_embed,
                img_feats,
                projection_mat=projection_mat,
                image_wh=image_wh,
                temp_instance_feature=temporal_memory,
            )
            cls_scores, bbox_preds = refine(
                query,
                cur_anchor,
                cur_anchor_embed,
                time_interval=time_interval,
            )
            all_cls_scores.append(cls_scores)
            all_bbox_preds.append(bbox_preds)
            cur_anchor = bbox_preds.detach()

            if (
                instance_bank_update is not None
                and self.num_single_frame_decoder > 0
                and layer_idx + 1 == self.num_single_frame_decoder
            ):
                query, cur_anchor = instance_bank_update(query, cur_anchor, cls_scores)
                cur_anchor = cur_anchor.detach()

            if layer_idx != self.num_layers - 1 and anchor_encoder is not None:
                cur_anchor_embed = anchor_encoder(cur_anchor)

        return (
            torch.stack(all_cls_scores, dim=0),
            torch.stack(all_bbox_preds, dim=0),
            query,
            cur_anchor,
        )


class SparseBox3DDecoderLite(nn.Module):
    """NMS-free top-k decode mirroring Sparse4D decoder contracts."""

    def __init__(
        self,
        max_detections: int = 100,
        *,
        score_threshold: float | None = None,
        sorted: bool = True,
    ) -> None:
        super().__init__()
        self.max_detections = int(max_detections)
        self.score_threshold = score_threshold
        self.sorted = bool(sorted)

    @staticmethod
    def _decode_box(box: torch.Tensor) -> torch.Tensor:
        code_size = box.shape[-1]
        x = box[..., 0]
        y = box[..., 1]
        z = box[..., 2]
        w = box[..., 3].exp() if code_size > 3 else torch.zeros_like(x)
        l = box[..., 4].exp() if code_size > 4 else torch.zeros_like(x)
        h = box[..., 5].exp() if code_size > 5 else torch.zeros_like(x)
        if code_size > COS_YAW:
            yaw = torch.atan2(box[..., SIN_YAW], box[..., COS_YAW])
        elif code_size > SIN_YAW:
            yaw = box[..., SIN_YAW]
        else:
            yaw = torch.zeros_like(x)
        vx = box[..., VX] if code_size > VX else torch.zeros_like(x)
        vy = box[..., VX + 1] if code_size > VX + 1 else torch.zeros_like(x)
        vz = box[..., VX + 2] if code_size > VX + 2 else torch.zeros_like(x)
        return torch.stack([x, y, z, w, l, h, yaw, vx, vy, vz], dim=-1)

    def decode(
        self,
        cls_scores: torch.Tensor,
        box_preds: torch.Tensor,
        instance_id: torch.Tensor | None = None,
        quality: torch.Tensor | None = None,
        *,
        output_idx: int = -1,
    ) -> list[dict[str, torch.Tensor]]:
        squeeze_cls = instance_id is not None

        if cls_scores.dim() == 4:
            cls_scores = cls_scores[output_idx]
        if box_preds.dim() == 4:
            box_preds = box_preds[output_idx]
        if quality is not None and quality.dim() == 4:
            quality = quality[output_idx]
        if cls_scores.dim() != 3 or box_preds.dim() != 3:
            raise ValueError("Expected cls_scores/box_preds with shape [B, Q, C] or [L, B, Q, C].")

        cls_scores = cls_scores.sigmoid()
        if squeeze_cls:
            cls_scores, cls_ids = cls_scores.max(dim=-1)
            cls_scores = cls_scores.unsqueeze(-1)
        else:
            cls_ids = None

        batch_size, num_pred, num_cls = cls_scores.shape
        topk = min(self.max_detections, num_pred * num_cls)
        topk_scores, indices = cls_scores.flatten(start_dim=1).topk(
            topk,
            dim=1,
            sorted=self.sorted,
        )
        if not squeeze_cls:
            cls_ids = indices % num_cls
        query_indices = indices // num_cls
        if self.score_threshold is not None:
            score_mask = topk_scores >= self.score_threshold
        else:
            score_mask = None

        if quality is not None:
            centerness = quality[..., CNS]
            centerness = torch.gather(centerness, 1, query_indices)
            cls_scores_origin = topk_scores.clone()
            topk_scores = topk_scores * centerness.sigmoid()
            topk_scores, sort_idx = torch.sort(topk_scores, dim=1, descending=True)
            indices = torch.gather(indices, 1, sort_idx)
            query_indices = torch.gather(query_indices, 1, sort_idx)
            if cls_ids is not None:
                cls_ids = torch.gather(cls_ids, 1, sort_idx)
            if score_mask is not None:
                score_mask = torch.gather(score_mask, 1, sort_idx)
        else:
            cls_scores_origin = None

        output: list[dict[str, torch.Tensor]] = []
        for batch_idx in range(batch_size):
            category_ids = cls_ids[batch_idx]
            if squeeze_cls:
                category_ids = category_ids[query_indices[batch_idx]]
            scores = topk_scores[batch_idx]
            box = box_preds[batch_idx, query_indices[batch_idx]]
            if score_mask is not None:
                valid_mask = score_mask[batch_idx]
                category_ids = category_ids[valid_mask]
                scores = scores[valid_mask]
                box = box[valid_mask]
            decoded_box = self._decode_box(box)
            result: dict[str, torch.Tensor] = {
                "boxes_3d": decoded_box.cpu(),
                "scores_3d": scores.cpu(),
                "labels_3d": category_ids.to(dtype=torch.long).cpu(),
            }
            if cls_scores_origin is not None:
                raw_scores = cls_scores_origin[batch_idx]
                if score_mask is not None:
                    raw_scores = raw_scores[score_mask[batch_idx]]
                result["cls_scores"] = raw_scores.cpu()
            if instance_id is not None:
                ids = instance_id[batch_idx, indices[batch_idx]]
                if score_mask is not None:
                    ids = ids[score_mask[batch_idx]]
                result["instance_ids"] = ids.cpu()
            output.append(result)
        return output
