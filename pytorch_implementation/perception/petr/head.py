"""PETR detection head in pure PyTorch (forward-only)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import PETRForwardConfig
from .postprocess import NMSFreeCoderLite
from .transformer import PETRTransformerLite
from .utils import (
    SinePositionalEncoding2D,
    build_img2lidars,
    inverse_sigmoid,
    pos2posemb3d,
    validate_petr_img_metas,
)


class PETRHeadLite(nn.Module):
    """Standalone PETR head that mirrors the reference forward contract."""

    def __init__(self, cfg: PETRForwardConfig, transformer: PETRTransformerLite) -> None:
        super().__init__()
        self.cfg = cfg
        self.transformer = transformer
        self.embed_dims = cfg.embed_dims
        self.num_classes = cfg.num_classes
        self.num_query = cfg.num_queries
        self.code_size = cfg.code_size
        self.depth_num = cfg.depth_num
        self.depth_start = cfg.depth_start
        self.position_dim = 3 * self.depth_num
        self.position_range = cfg.position_range
        self.lidar_discretization = cfg.lidar_discretization
        self.position_level = 0

        self.input_proj = nn.Conv2d(cfg.backbone_neck.out_channels, self.embed_dims, kernel_size=1)
        self.position_encoder = nn.Sequential(
            nn.Conv2d(self.position_dim, self.embed_dims * 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1),
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
        self.cls_branches, self.reg_branches = self._build_branches(cfg.num_decoder_layers)

        self.bbox_coder = NMSFreeCoderLite(
            pc_range=cfg.pc_range,
            post_center_range=cfg.post_center_range,
            max_num=cfg.max_num,
            num_classes=cfg.num_classes,
            score_threshold=cfg.score_threshold,
        )
        nn.init.uniform_(self.reference_points.weight, 0.0, 1.0)

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
        # PETR keeps one shared prediction branch per decoder stage.
        return (
            nn.ModuleList([cls_branch for _ in range(num_pred)]),
            nn.ModuleList([reg_branch for _ in range(num_pred)]),
        )

    @staticmethod
    def _shape_hw(shape: object, *, field_name: str, batch_idx: int, cam_idx: int) -> tuple[int, int]:
        if not isinstance(shape, (list, tuple)) or len(shape) < 2:
            raise ValueError(
                f"img_metas[{batch_idx}]['{field_name}'][{cam_idx}] must provide at least (H, W), got {shape!r}."
            )
        return int(shape[0]), int(shape[1])

    def _build_img_masks(
        self,
        img_metas: list[dict],
        *,
        batch_size: int,
        num_cams: int,
        device: torch.device,
    ) -> torch.Tensor:
        validate_petr_img_metas(img_metas, batch_size=batch_size, num_cams=num_cams)
        pad_shapes = img_metas[0].get("pad_shape", img_metas[0]["img_shape"])
        pad_h, pad_w = self._shape_hw(pad_shapes[0], field_name="pad_shape", batch_idx=0, cam_idx=0)

        masks = torch.ones((batch_size, num_cams, pad_h, pad_w), device=device, dtype=torch.bool)
        for batch_idx in range(batch_size):
            img_shapes = img_metas[batch_idx]["img_shape"]
            for cam_idx in range(num_cams):
                img_h, img_w = self._shape_hw(img_shapes[cam_idx], field_name="img_shape", batch_idx=batch_idx, cam_idx=cam_idx)
                masks[batch_idx, cam_idx, :img_h, :img_w] = False
        return masks

    def position_embeding(
        self,
        img_feats: list[torch.Tensor],
        img_metas: list[dict],
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build PETR 3D positional embeddings from camera intrinsics/extrinsics."""

        eps = 1e-5
        feat = img_feats[self.position_level]
        batch_size, num_cams, _, height, width = feat.shape
        pad_shapes = img_metas[0].get("pad_shape", img_metas[0]["img_shape"])
        pad_h, pad_w = self._shape_hw(pad_shapes[0], field_name="pad_shape", batch_idx=0, cam_idx=0)

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

        depth_bins = coords_d.shape[0]
        coords = torch.stack(
            torch.meshgrid(coords_w, coords_h, coords_d, indexing="ij"),
            dim=-1,
        )  # [W, H, D, 3]
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), dim=-1)  # [W, H, D, 4]
        coords[..., :2] = coords[..., :2] * torch.maximum(
            coords[..., 2:3],
            torch.ones_like(coords[..., 2:3]) * eps,
        )

        img2lidars = build_img2lidars(img_metas, device=feat.device, dtype=feat.dtype, num_cams=num_cams)
        coords = coords.view(1, 1, width, height, depth_bins, 4, 1).repeat(batch_size, num_cams, 1, 1, 1, 1, 1)
        img2lidars = img2lidars.view(batch_size, num_cams, 1, 1, 1, 4, 4).repeat(1, 1, width, height, depth_bins, 1, 1)
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

        coords_mask = ((coords3d > 1.0) | (coords3d < 0.0)).flatten(-2).sum(-1) > (depth_bins * 0.5)
        coords_mask = masks | coords_mask.permute(0, 1, 3, 2)

        coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(batch_size * num_cams, -1, height, width)
        coords3d = inverse_sigmoid(coords3d)
        coords_pos_embed = self.position_encoder(coords3d)
        coords_pos_embed = coords_pos_embed.view(batch_size, num_cams, self.embed_dims, height, width)
        return coords_pos_embed, coords_mask

    def forward(self, mlvl_feats: list[torch.Tensor], img_metas: list[dict]) -> dict[str, torch.Tensor]:
        if not mlvl_feats:
            raise ValueError("mlvl_feats must contain at least one feature level.")
        x = mlvl_feats[self.position_level]
        if x.dim() != 5:
            raise ValueError(
                f"PETRHeadLite expects mlvl_feats[{self.position_level}] shaped [B, Ncam, C, H, W], got {tuple(x.shape)}."
            )
        batch_size, num_cams = x.shape[:2]
        validate_petr_img_metas(img_metas, batch_size=batch_size, num_cams=num_cams)
        masks = self._build_img_masks(img_metas, batch_size=batch_size, num_cams=num_cams, device=x.device)

        x = self.input_proj(x.flatten(0, 1)).view(batch_size, num_cams, self.embed_dims, x.shape[-2], x.shape[-1])
        masks = F.interpolate(masks.float(), size=x.shape[-2:]).to(torch.bool)

        coords_position_embeding, _ = self.position_embeding(mlvl_feats, img_metas, masks)
        pos_embeds = [self.positional_encoding(masks[:, cam_idx]).unsqueeze(1) for cam_idx in range(num_cams)]
        sin_embed = torch.cat(pos_embeds, dim=1)
        sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view_as(x)
        pos_embed = coords_position_embeding + sin_embed

        query_ids = torch.arange(self.num_query, device=x.device)
        reference_points = self.reference_points(query_ids)
        query_embeds = self.query_embedding(pos2posemb3d(reference_points, num_pos_feats=self.embed_dims // 2))
        reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)

        outs_dec, _ = self.transformer(x, masks, query_embeds, pos_embed, reg_branch=self.reg_branches)
        outs_dec = torch.nan_to_num(outs_dec)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())
            if reference.shape[-1] != 3:
                raise AssertionError(f"Expected 3D reference points, got {reference.shape[-1]}.")
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            tmp = self.reg_branches[lvl](outs_dec[lvl])
            tmp[..., 0:2] = (tmp[..., 0:2] + reference[..., 0:2]).sigmoid()
            tmp[..., 4:5] = (tmp[..., 4:5] + reference[..., 2:3]).sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(tmp)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        all_bbox_preds[..., 0:1] = (
            all_bbox_preds[..., 0:1] * (self.cfg.pc_range[3] - self.cfg.pc_range[0]) + self.cfg.pc_range[0]
        )
        all_bbox_preds[..., 1:2] = (
            all_bbox_preds[..., 1:2] * (self.cfg.pc_range[4] - self.cfg.pc_range[1]) + self.cfg.pc_range[1]
        )
        all_bbox_preds[..., 4:5] = (
            all_bbox_preds[..., 4:5] * (self.cfg.pc_range[5] - self.cfg.pc_range[2]) + self.cfg.pc_range[2]
        )

        return {
            "all_cls_scores": all_cls_scores,
            "all_bbox_preds": all_bbox_preds,
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
        }

    def get_bboxes(self, preds_dicts: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        return self.bbox_coder.decode(preds_dicts)

