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


def _reshape_projection(
    projection_mat: torch.Tensor,
    *,
    batch_size: int,
    num_cams: int,
) -> torch.Tensor:
    if projection_mat.dim() != 4:
        raise ValueError("projection_mat must have shape [B, Ncam, 3/4, 4].")
    if projection_mat.shape[0] == 1 and batch_size > 1:
        projection_mat = projection_mat.expand(batch_size, -1, -1, -1)
    if projection_mat.shape[0] != batch_size or projection_mat.shape[1] != num_cams:
        raise ValueError("projection_mat batch/camera dimensions do not match image features.")
    if projection_mat.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError("projection_mat must have trailing shape [3, 4] or [4, 4].")
    return projection_mat


def _reshape_image_wh(
    image_wh: torch.Tensor,
    *,
    batch_size: int,
    num_cams: int,
) -> torch.Tensor:
    if image_wh.dim() != 3 or image_wh.shape[-1] != 2:
        raise ValueError("image_wh must have shape [B, Ncam, 2].")
    if image_wh.shape[0] == 1 and batch_size > 1:
        image_wh = image_wh.expand(batch_size, -1, -1)
    if image_wh.shape[0] != batch_size or image_wh.shape[1] != num_cams:
        raise ValueError("image_wh batch/camera dimensions do not match image features.")
    return image_wh


def _normalize_timestamp(
    timestamp: Any,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if timestamp is None:
        return None
    value = torch.as_tensor(timestamp, device=device, dtype=dtype).reshape(-1)
    if value.numel() == 1:
        value = value.expand(batch_size)
    if value.numel() != batch_size:
        return None
    return value


def _meta_to_tensors(
    metas: Any,
    *,
    batch_size: int,
    num_cams: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    projection_mat = None
    image_wh = None
    timestamp = None
    img_metas: list[dict[str, Any]] | None = None

    if isinstance(metas, dict):
        projection_mat = metas.get("projection_mat", metas.get("lidar2img"))
        image_wh = metas.get("image_wh")
        timestamp = metas.get("timestamp")
        raw_img_metas = metas.get("img_metas")
        if isinstance(raw_img_metas, list) and all(isinstance(x, dict) for x in raw_img_metas):
            img_metas = raw_img_metas
    elif isinstance(metas, list):
        projection_items = []
        wh_items = []
        timestamp_items = []
        if all(isinstance(x, dict) for x in metas):
            img_metas = metas
        for sample_meta in metas:
            if not isinstance(sample_meta, dict):
                continue
            lidar2img = sample_meta.get("projection_mat", sample_meta.get("lidar2img"))
            if lidar2img is not None:
                projection_items.append(torch.as_tensor(lidar2img, dtype=dtype))
            sample_wh = sample_meta.get("image_wh")
            if sample_wh is not None:
                wh_items.append(torch.as_tensor(sample_wh, dtype=dtype))
            else:
                img_shape = sample_meta.get("img_shape")
                if img_shape is not None:
                    cur_wh = []
                    for shape in img_shape:
                        height = float(shape[0])
                        width = float(shape[1])
                        cur_wh.append([width, height])
                    wh_items.append(torch.tensor(cur_wh, dtype=dtype))
            if "timestamp" in sample_meta:
                timestamp_items.append(sample_meta["timestamp"])
        if projection_items:
            projection_mat = torch.stack(projection_items, dim=0)
        if wh_items:
            image_wh = torch.stack(wh_items, dim=0)
        if timestamp_items:
            timestamp = timestamp_items

    if projection_mat is None:
        projection_mat = _default_projection(batch_size, num_cams, device=device, dtype=dtype)
    else:
        projection_mat = torch.as_tensor(projection_mat, dtype=dtype, device=device)
        projection_mat = _reshape_projection(projection_mat, batch_size=batch_size, num_cams=num_cams)
    if image_wh is None:
        image_wh = _default_image_wh(batch_size, num_cams, device=device, dtype=dtype)
    else:
        image_wh = torch.as_tensor(image_wh, dtype=dtype, device=device)
        image_wh = _reshape_image_wh(image_wh, batch_size=batch_size, num_cams=num_cams)

    bank_metas: dict[str, Any] = {}
    normalized_timestamp = _normalize_timestamp(
        timestamp,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )
    if normalized_timestamp is not None:
        bank_metas["timestamp"] = normalized_timestamp
    if img_metas is not None:
        bank_metas["img_metas"] = img_metas

    return projection_mat, image_wh, bank_metas


class Sparse4DHeadLite(nn.Module):
    """Minimal Sparse4D head: instance bank + decoder + box decode."""

    def __init__(self, cfg: Sparse4DForwardConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.instance_bank = InstanceBankLite(
            num_queries=cfg.num_queries,
            embed_dims=cfg.embed_dims,
            box_code_size=cfg.box_code_size,
            num_temp_instances=cfg.num_temp_instances,
            default_time_interval=cfg.default_time_interval,
            confidence_decay=cfg.confidence_decay,
            max_time_interval=cfg.max_time_interval,
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
            num_levels=cfg.backbone_neck.num_outs,
            num_cams=cfg.num_cams,
            num_single_frame_decoder=cfg.num_single_frame_decoder,
            normalize_yaw=cfg.normalize_yaw,
            refine_yaw=cfg.refine_yaw,
        )
        self.box_decoder = SparseBox3DDecoderLite(
            max_detections=cfg.max_detections,
            score_threshold=cfg.score_threshold,
            sorted=cfg.sorted_decoding,
        )

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
        projection_mat, image_wh, bank_metas = _meta_to_tensors(
            metas,
            batch_size=batch_size,
            num_cams=self.cfg.num_cams,
            device=device,
            dtype=dtype,
        )
        instance_feature, anchors, temp_instance_feature, _, time_interval = self.instance_bank(
            batch_size,
            metas=bank_metas,
            device=device,
            dtype=dtype,
        )
        anchor_embed = self.anchor_encoder(anchors)
        all_cls_scores, all_bbox_preds, final_instance_feature, final_anchor = self.decoder(
            instance_feature,
            anchor_embed,
            anchors,
            mlvl_feats,
            projection_mat=projection_mat,
            image_wh=image_wh,
            time_interval=time_interval,
            temp_instance_feature=temp_instance_feature,
            anchor_encoder=self.anchor_encoder,
            instance_bank_update=(
                self.instance_bank.update if self.instance_bank.num_temp_instances > 0 else None
            ),
        )
        self.instance_bank.cache(
            final_instance_feature,
            final_anchor,
            all_cls_scores[-1],
            metas=bank_metas,
            feature_maps=mlvl_feats,
        )
        return {
            "all_cls_scores": all_cls_scores,
            "all_bbox_preds": all_bbox_preds,
        }

    def get_bboxes(self, outputs: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        return self.box_decoder.decode(
            outputs["all_cls_scores"],
            outputs["all_bbox_preds"],
            output_idx=-1,
        )
