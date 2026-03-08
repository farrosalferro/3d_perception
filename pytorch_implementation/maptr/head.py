"""MapTR detection head in pure PyTorch (forward-only)."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from .config import MapTRForwardConfig
from .postprocess import MapTRNMSFreeCoderLite
from .transformer import MapTRPerceptionTransformerLite
from .utils import SinePositionalEncoding2D, points_to_boxes


class MapTRHeadLite(nn.Module):
    """Standalone MapTR head with BEV tokens and vectorized point queries."""

    def __init__(self, cfg: MapTRForwardConfig, transformer: MapTRPerceptionTransformerLite) -> None:
        super().__init__()
        self.cfg = cfg
        self.transformer = transformer
        self.embed_dims = cfg.embed_dims
        self.num_classes = cfg.num_map_classes
        self.num_vec = cfg.num_vec
        self.num_pts_per_vec = cfg.num_pts_per_vec
        self.num_query = cfg.num_query
        self.code_size = cfg.code_size
        self.bev_h = cfg.bev_h
        self.bev_w = cfg.bev_w
        self.position_level = 0

        self.input_proj = nn.Conv2d(cfg.backbone_neck.out_channels, self.embed_dims, kernel_size=1)
        self.positional_encoding = SinePositionalEncoding2D(self.embed_dims // 2, normalize=True)
        self.adapt_pos3d = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1),
        )

        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
        self.instance_embedding = nn.Embedding(self.num_vec, self.embed_dims)
        self.pts_embedding = nn.Embedding(self.num_pts_per_vec, self.embed_dims)
        self.cls_branches, self.reg_branches = self._build_branches(cfg.num_decoder_layers)

        self.bbox_coder = MapTRNMSFreeCoderLite(
            post_center_range=cfg.post_center_range,
            pc_range=cfg.pc_range,
            max_num=cfg.max_num,
            num_classes=cfg.num_map_classes,
            score_threshold=cfg.score_threshold,
        )

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

    def _build_query_embedding(self, device: torch.device) -> torch.Tensor:
        instance_ids = torch.arange(self.num_vec, device=device)
        pts_ids = torch.arange(self.num_pts_per_vec, device=device)
        instance_embed = self.instance_embedding(instance_ids)
        pts_embed = self.pts_embedding(pts_ids)
        return (instance_embed[:, None, :] + pts_embed[None, :, :]).reshape(self.num_query, self.embed_dims)

    def forward(self, mlvl_feats: list[torch.Tensor], img_metas: list[dict]) -> dict[str, torch.Tensor]:
        x = mlvl_feats[self.position_level]
        batch_size, num_cams = x.shape[:2]
        masks = self._build_img_masks(img_metas, batch_size=batch_size, num_cams=num_cams, device=x.device)

        x = self.input_proj(x.flatten(0, 1)).view(batch_size, num_cams, self.embed_dims, x.shape[-2], x.shape[-1])
        masks = F.interpolate(masks.float(), size=x.shape[-2:], mode="nearest").to(torch.bool)

        pos_embeds = [self.positional_encoding(masks[:, cam_idx]).unsqueeze(1) for cam_idx in range(num_cams)]
        cam_pos_embed = torch.cat(pos_embeds, dim=1)
        cam_pos_embed = self.adapt_pos3d(cam_pos_embed.flatten(0, 1)).view_as(x)

        query_embed = self._build_query_embedding(device=x.device)
        bev_ids = torch.arange(self.bev_h * self.bev_w, device=x.device)
        bev_queries = self.bev_embedding(bev_ids)
        bev_mask = torch.zeros((batch_size, self.bev_h, self.bev_w), device=x.device, dtype=torch.bool)
        bev_pos = self.positional_encoding(bev_mask).flatten(2).transpose(1, 2)

        bev_embed, outs_dec = self.transformer(
            x,
            masks,
            query_embed,
            cam_pos_embed,
            bev_queries,
            bev_pos,
        )
        outs_dec = torch.nan_to_num(outs_dec)
        outs_dec = outs_dec.permute(0, 2, 1, 3).contiguous()  # [L, B, Q, C]

        outputs_classes = []
        outputs_coords = []
        outputs_pts_coords = []
        for lvl in range(outs_dec.shape[0]):
            hidden = outs_dec[lvl]
            vec_embedding = hidden.reshape(batch_size, self.num_vec, self.num_pts_per_vec, -1).mean(2)
            outputs_class = self.cls_branches[lvl](vec_embedding)
            pts = self.reg_branches[lvl](hidden).sigmoid().reshape(batch_size, self.num_vec, self.num_pts_per_vec, 2)
            outputs_classes.append(outputs_class)
            outputs_pts_coords.append(pts)
            outputs_coords.append(points_to_boxes(pts))

        return {
            "bev_embed": bev_embed,
            "all_cls_scores": torch.stack(outputs_classes),
            "all_bbox_preds": torch.stack(outputs_coords),
            "all_pts_preds": torch.stack(outputs_pts_coords),
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
            "enc_pts_preds": None,
        }

    def get_bboxes(self, preds_dicts: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        return self.bbox_coder.decode(preds_dicts)
