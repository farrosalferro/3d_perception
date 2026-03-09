"""Sparse4D head assembly in pure PyTorch."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .blocks import SparseBox3DEncoderLite
from .config import Sparse4DForwardConfig
from .decoder import Sparse4DDecoderLite, SparseBox3DDecoderLite
from .instance_bank import InstanceBankLite


def _default_projection(
    batch_size: int,
    num_cams: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    eye = torch.eye(4, device=device, dtype=dtype).view(1, 1, 4, 4)
    return eye.repeat(batch_size, num_cams, 1, 1)


def _default_image_wh(
    batch_size: int,
    num_cams: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.ones(batch_size, num_cams, 2, device=device, dtype=dtype)


def _meta_to_tensors(
    metas: Any,
    *,
    batch_size: int,
    num_cams: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    projection_mat = None
    image_wh = None

    if isinstance(metas, dict):
        projection_mat = metas.get("projection_mat")
        image_wh = metas.get("image_wh")
    elif isinstance(metas, list):
        projection_items = []
        wh_items = []
        for sample_meta in metas:
            lidar2img = sample_meta.get("lidar2img")
            if lidar2img is not None:
                projection_items.append(torch.as_tensor(lidar2img, dtype=dtype))
            img_shape = sample_meta.get("img_shape")
            if img_shape is not None:
                cur_wh = []
                for shape in img_shape:
                    height = float(shape[0])
                    width = float(shape[1])
                    cur_wh.append([width, height])
                wh_items.append(torch.tensor(cur_wh, dtype=dtype))
        if projection_items:
            projection_mat = torch.stack(projection_items, dim=0)
        if wh_items:
            image_wh = torch.stack(wh_items, dim=0)

    if projection_mat is None:
        projection_mat = _default_projection(batch_size, num_cams, device=device, dtype=dtype)
    else:
        projection_mat = torch.as_tensor(projection_mat, dtype=dtype, device=device)
    if image_wh is None:
        image_wh = _default_image_wh(batch_size, num_cams, device=device, dtype=dtype)
    else:
        image_wh = torch.as_tensor(image_wh, dtype=dtype, device=device)

    return projection_mat, image_wh


class Sparse4DHeadLite(nn.Module):
    """Minimal Sparse4D head: instance bank + decoder + box decode."""

    def __init__(self, cfg: Sparse4DForwardConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.instance_bank = InstanceBankLite(
            num_queries=cfg.num_queries,
            embed_dims=cfg.embed_dims,
            box_code_size=cfg.box_code_size,
        )
        self.anchor_encoder = SparseBox3DEncoderLite(cfg.box_code_size, cfg.embed_dims)
        self.decoder = Sparse4DDecoderLite(
            num_layers=cfg.num_decoder_layers,
            embed_dims=cfg.embed_dims,
            ffn_dims=cfg.ffn_dims,
            num_heads=cfg.num_heads,
            num_classes=cfg.num_classes,
            box_code_size=cfg.box_code_size,
            dropout=cfg.dropout,
        )
        self.box_decoder = SparseBox3DDecoderLite(max_detections=cfg.max_detections)

    def forward(
        self,
        mlvl_feats: list[torch.Tensor],
        metas: Any = None,
    ) -> dict[str, torch.Tensor]:
        if not mlvl_feats:
            raise ValueError("mlvl_feats cannot be empty.")
        batch_size = mlvl_feats[0].shape[0]
        dtype = mlvl_feats[0].dtype
        device = mlvl_feats[0].device
        projection_mat, image_wh = _meta_to_tensors(
            metas,
            batch_size=batch_size,
            num_cams=self.cfg.num_cams,
            device=device,
            dtype=dtype,
        )
        instance_feature, anchors = self.instance_bank(batch_size, device=device, dtype=dtype)
        anchor_embed = self.anchor_encoder(anchors)
        all_cls_scores, all_bbox_preds = self.decoder(
            instance_feature,
            anchor_embed,
            anchors,
            mlvl_feats,
            projection_mat=projection_mat,
            image_wh=image_wh,
        )
        return {
            "all_cls_scores": all_cls_scores,
            "all_bbox_preds": all_bbox_preds,
        }

    def get_bboxes(self, outputs: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        cls_scores = outputs["all_cls_scores"][-1]
        bbox_preds = outputs["all_bbox_preds"][-1]
        return self.box_decoder.decode(cls_scores, bbox_preds)
