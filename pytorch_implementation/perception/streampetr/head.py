"""StreamPETR detection head in pure PyTorch (forward-only)."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from .config import StreamPETRForwardConfig
from .postprocess import NMSFreeCoderLite
from .transformer import StreamPETRTemporalTransformerLite
from .utils import (
    SinePositionalEncoding2D,
    inverse_sigmoid,
    memory_refresh,
    pos2posemb1d,
    pos2posemb3d,
    topk_gather,
)


class StreamPETRHeadLite(nn.Module):
    """Standalone StreamPETR head with object-centric temporal memory."""

    def __init__(self, cfg: StreamPETRForwardConfig, transformer: StreamPETRTemporalTransformerLite) -> None:
        super().__init__()
        self.cfg = cfg
        self.transformer = transformer
        self.embed_dims = cfg.embed_dims
        self.num_classes = cfg.num_classes
        self.num_query = cfg.num_queries
        self.code_size = cfg.code_size
        self.depth_num = cfg.depth_num
        self.depth_start = cfg.depth_start
        self.position_range = cfg.position_range
        self.lidar_discretization = cfg.lidar_discretization
        self.position_level = 0
        self.memory_len = cfg.memory_len
        self.topk_proposals = cfg.topk_proposals
        self.num_propagated = cfg.num_propagated

        self.input_proj = nn.Conv2d(cfg.backbone_neck.out_channels, self.embed_dims, kernel_size=1)
        self.position_encoder = nn.Sequential(
            nn.Conv2d(3 * self.depth_num, self.embed_dims * 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dims * 2, self.embed_dims, kernel_size=1),
        )
        self.positional_encoding = SinePositionalEncoding2D(self.embed_dims // 2, normalize=True)
        self.adapt_pos3d = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1),
        )

        self.reference_points = nn.Embedding(self.num_query, 3)
        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims * 3 // 2, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.cls_branches, self.reg_branches = self._build_branches(cfg.num_decoder_layers)

        self.bbox_coder = NMSFreeCoderLite(
            post_center_range=cfg.post_center_range,
            max_num=cfg.max_num,
            num_classes=cfg.num_classes,
            score_threshold=cfg.score_threshold,
        )
        nn.init.uniform_(self.reference_points.weight, 0.0, 1.0)
        self.reset_memory()

    def _build_branches(self, num_pred: int) -> tuple[nn.ModuleList, nn.ModuleList]:
        cls_branch = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.num_classes),
        )
        reg_branch = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.code_size),
        )
        return (
            nn.ModuleList([copy.deepcopy(cls_branch) for _ in range(num_pred)]),
            nn.ModuleList([copy.deepcopy(reg_branch) for _ in range(num_pred)]),
        )

    def reset_memory(self) -> None:
        self.memory_embedding: torch.Tensor | None = None
        self.memory_reference_point: torch.Tensor | None = None
        self.memory_timestamp: torch.Tensor | None = None

    def _build_img_masks(
        self,
        img_metas: list[dict],
        *,
        batch_size: int,
        num_cams: int,
        device: torch.device,
    ) -> torch.Tensor:
        pad_shapes = img_metas[0].get("pad_shape", img_metas[0].get("img_shape"))
        if pad_shapes is None:
            raise KeyError("img_metas must provide 'pad_shape' or 'img_shape'.")
        pad_h, pad_w = int(pad_shapes[0][0]), int(pad_shapes[0][1])

        masks = torch.ones((batch_size, num_cams, pad_h, pad_w), device=device, dtype=torch.bool)
        for batch_idx in range(batch_size):
            img_shapes = img_metas[batch_idx].get("img_shape")
            if img_shapes is None:
                raise KeyError("img_metas entries must provide 'img_shape'.")
            for cam_idx in range(num_cams):
                img_h, img_w = int(img_shapes[cam_idx][0]), int(img_shapes[cam_idx][1])
                masks[batch_idx, cam_idx, :img_h, :img_w] = False
        return masks

    def pre_update_memory(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        prev_exists: torch.Tensor | None,
    ) -> None:
        if self.memory_embedding is None:
            self.memory_embedding = torch.zeros(batch_size, self.memory_len, self.embed_dims, device=device, dtype=dtype)
            self.memory_reference_point = torch.zeros(batch_size, self.memory_len, 3, device=device, dtype=dtype)
            self.memory_timestamp = torch.zeros(batch_size, self.memory_len, 1, device=device, dtype=dtype)

        if prev_exists is None:
            prev_exists = torch.ones(batch_size, device=device, dtype=dtype)
        else:
            prev_exists = prev_exists.to(device=device, dtype=dtype).view(batch_size)

        self.memory_embedding = memory_refresh(self.memory_embedding, prev_exists)
        self.memory_reference_point = memory_refresh(self.memory_reference_point, prev_exists)
        self.memory_timestamp = memory_refresh(self.memory_timestamp, prev_exists)
        self.memory_timestamp = self.memory_timestamp + prev_exists.view(batch_size, 1, 1)

    def post_update_memory(
        self,
        all_cls_scores: torch.Tensor,
        all_bbox_preds: torch.Tensor,
        outs_dec_main: torch.Tensor,
    ) -> None:
        if self.memory_embedding is None or self.memory_reference_point is None or self.memory_timestamp is None:
            return

        rec_memory = outs_dec_main[-1]  # [B, Q, C]
        rec_reference_points = all_bbox_preds[-1][..., :3]  # [B, Q, 3] in metric space
        rec_score = all_cls_scores[-1].sigmoid().topk(1, dim=-1).values[..., 0]  # [B, Q]
        k = min(self.topk_proposals, rec_score.shape[1], self.memory_len)
        if k <= 0:
            return
        _, topk_indexes = torch.topk(rec_score, k, dim=1)

        rec_memory = topk_gather(rec_memory, topk_indexes).detach()
        rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
        rec_timestamp = torch.zeros(
            rec_memory.shape[0],
            rec_memory.shape[1],
            1,
            dtype=rec_memory.dtype,
            device=rec_memory.device,
        )

        pc_min = rec_reference_points.new_tensor(self.cfg.pc_range[:3]).view(1, 1, 3)
        pc_max = rec_reference_points.new_tensor(self.cfg.pc_range[3:6]).view(1, 1, 3)
        rec_reference_points = (rec_reference_points - pc_min) / torch.clamp(pc_max - pc_min, min=1e-5)
        rec_reference_points = rec_reference_points.clamp(0.0, 1.0)

        self.memory_embedding = torch.cat([rec_memory, self.memory_embedding], dim=1)[:, : self.memory_len]
        self.memory_reference_point = torch.cat([rec_reference_points, self.memory_reference_point], dim=1)[:, : self.memory_len]
        self.memory_timestamp = torch.cat([rec_timestamp, self.memory_timestamp], dim=1)[:, : self.memory_len]

    def position_embeding(
        self,
        feat: torch.Tensor,
        img_metas: list[dict],
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build StreamPETR-style 3D positional embeddings from camera geometry."""

        eps = 1e-5
        batch_size, num_cams, _, height, width = feat.shape
        pad_shapes = img_metas[0].get("pad_shape", img_metas[0].get("img_shape"))
        pad_h, pad_w = int(pad_shapes[0][0]), int(pad_shapes[0][1])

        coords_h = torch.arange(height, device=feat.device, dtype=feat.dtype) * float(pad_h) / float(height)
        coords_w = torch.arange(width, device=feat.device, dtype=feat.dtype) * float(pad_w) / float(width)
        depth_idx = torch.arange(self.depth_num, device=feat.device, dtype=feat.dtype)
        if self.lidar_discretization:
            depth_idx_1 = depth_idx + 1.0
            bin_size = (self.position_range[3] - self.depth_start) / (
                self.depth_num * (1.0 + self.depth_num)
            )
            coords_d = self.depth_start + bin_size * depth_idx * depth_idx_1
        else:
            bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * depth_idx

        coords = torch.stack(
            torch.meshgrid(coords_w, coords_h, coords_d, indexing="ij"),
            dim=-1,
        )  # [W, H, D, 3]
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), dim=-1)  # [W, H, D, 4]
        coords[..., :2] = coords[..., :2] * torch.maximum(
            coords[..., 2:3],
            torch.ones_like(coords[..., 2:3]) * eps,
        )

        img2lidars = []
        for meta in img_metas:
            lidar2img = meta.get("lidar2img")
            if lidar2img is None:
                raise KeyError("img_metas entries must provide 'lidar2img'.")
            mats = [torch.as_tensor(mat, device=feat.device, dtype=feat.dtype) for mat in lidar2img]
            stacked = torch.stack(mats, dim=0)
            img2lidars.append(torch.linalg.inv(stacked))
        img2lidars = torch.stack(img2lidars, dim=0)  # [B, Ncam, 4, 4]

        coords = coords.view(1, 1, width, height, self.depth_num, 4, 1).repeat(batch_size, num_cams, 1, 1, 1, 1, 1)
        img2lidars = img2lidars.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]  # [B, Ncam, W, H, D, 3]
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / (
            self.position_range[3] - self.position_range[0]
        )
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / (
            self.position_range[4] - self.position_range[1]
        )
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (
            self.position_range[5] - self.position_range[2]
        )

        coords_mask = ((coords3d > 1.0) | (coords3d < 0.0)).flatten(-2).sum(-1) > (self.depth_num * 0.5)
        coords_mask = masks | coords_mask.permute(0, 1, 3, 2)

        coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(batch_size * num_cams, -1, height, width)
        coords3d = inverse_sigmoid(coords3d)
        coords_pos_embed = self.position_encoder(coords3d).view(batch_size, num_cams, self.embed_dims, height, width)
        return coords_pos_embed, coords_mask

    def temporal_alignment(
        self,
        query_pos: torch.Tensor,
        tgt: torch.Tensor,
        reference_points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Append propagated memory queries and prepare temporal memory keys."""

        if self.memory_embedding is None or self.memory_reference_point is None or self.memory_timestamp is None:
            return tgt, query_pos, reference_points, None, None

        num_propagated = min(self.num_propagated, self.memory_embedding.shape[1])
        propagated_memory = self.memory_embedding[:, :num_propagated]
        propagated_ref = self.memory_reference_point[:, :num_propagated]
        if num_propagated > 0:
            propagated_pos = self.query_embedding(
                pos2posemb3d(propagated_ref, num_pos_feats=self.embed_dims // 2)
            )
            propagated_time = self.time_embedding(
                pos2posemb1d(
                    self.memory_timestamp[:, :num_propagated, 0],
                    num_pos_feats=self.embed_dims // 2,
                )
            )
            propagated_pos = propagated_pos + propagated_time

            tgt = torch.cat([tgt, propagated_memory], dim=1)
            query_pos = torch.cat([query_pos, propagated_pos], dim=1)
            reference_points = torch.cat([reference_points, propagated_ref], dim=1)

        temporal_memory = self.memory_embedding[:, num_propagated:]
        temporal_ref = self.memory_reference_point[:, num_propagated:]
        temporal_time = self.memory_timestamp[:, num_propagated:, 0]
        if temporal_memory.numel() == 0:
            return tgt, query_pos, reference_points, None, None

        temporal_pos = self.query_embedding(pos2posemb3d(temporal_ref, num_pos_feats=self.embed_dims // 2))
        temporal_pos = temporal_pos + self.time_embedding(
            pos2posemb1d(temporal_time, num_pos_feats=self.embed_dims // 2)
        )
        return tgt, query_pos, reference_points, temporal_memory, temporal_pos

    def forward(self, mlvl_feats: list[torch.Tensor], img_metas: list[dict], **data) -> dict[str, torch.Tensor]:
        x = mlvl_feats[self.position_level]
        batch_size, num_cams = x.shape[:2]
        prev_exists = data.get("prev_exists")
        if prev_exists is not None and not torch.is_tensor(prev_exists):
            prev_exists = torch.as_tensor(prev_exists, dtype=x.dtype, device=x.device)
        self.pre_update_memory(
            batch_size=batch_size,
            device=x.device,
            dtype=x.dtype,
            prev_exists=prev_exists,
        )

        masks = self._build_img_masks(img_metas, batch_size=batch_size, num_cams=num_cams, device=x.device)
        x = self.input_proj(x.flatten(0, 1)).view(batch_size, num_cams, self.embed_dims, x.shape[-2], x.shape[-1])
        masks = F.interpolate(masks.float(), size=x.shape[-2:], mode="nearest").to(torch.bool)

        coords_position_embeding, masks = self.position_embeding(x, img_metas, masks)
        pos_embeds = [self.positional_encoding(masks[:, cam_idx]).unsqueeze(1) for cam_idx in range(num_cams)]
        sin_embed = torch.cat(pos_embeds, dim=1)
        sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view_as(x)
        pos_embed = coords_position_embeding + sin_embed

        query_ids = torch.arange(self.num_query, device=x.device)
        reference_points = self.reference_points(query_ids)
        query_pos = self.query_embedding(pos2posemb3d(reference_points, num_pos_feats=self.embed_dims // 2))
        query_pos = query_pos.unsqueeze(0).repeat(batch_size, 1, 1)
        reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
        tgt = torch.zeros_like(query_pos)

        tgt, query_pos, reference_points, temporal_memory, temporal_pos = self.temporal_alignment(
            query_pos, tgt, reference_points
        )
        outs_dec, _ = self.transformer(
            x,
            masks,
            query_pos,
            pos_embed,
            tgt=tgt,
            temp_memory=temporal_memory,
            temp_pos=temporal_pos,
        )
        outs_dec = torch.nan_to_num(outs_dec)
        outs_dec = outs_dec.permute(0, 2, 1, 3)  # [L, B, Q_total, C]
        outs_dec_main = outs_dec[:, :, : self.num_query, :]
        reference = inverse_sigmoid(reference_points[:, : self.num_query, :].clone())

        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec_main.shape[0]):
            outputs_class = self.cls_branches[lvl](outs_dec_main[lvl])
            tmp = self.reg_branches[lvl](outs_dec_main[lvl])
            tmp[..., 0:3] = (tmp[..., 0:3] + reference[..., 0:3]).sigmoid()
            tmp[..., 0:1] = tmp[..., 0:1] * (self.cfg.pc_range[3] - self.cfg.pc_range[0]) + self.cfg.pc_range[0]
            tmp[..., 1:2] = tmp[..., 1:2] * (self.cfg.pc_range[4] - self.cfg.pc_range[1]) + self.cfg.pc_range[1]
            tmp[..., 2:3] = tmp[..., 2:3] * (self.cfg.pc_range[5] - self.cfg.pc_range[2]) + self.cfg.pc_range[2]
            outputs_classes.append(outputs_class)
            outputs_coords.append(tmp)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        self.post_update_memory(all_cls_scores, all_bbox_preds, outs_dec_main)

        return {
            "all_cls_scores": all_cls_scores,
            "all_bbox_preds": all_bbox_preds,
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
        }

    def get_bboxes(self, preds_dicts: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        return self.bbox_coder.decode(preds_dicts)

