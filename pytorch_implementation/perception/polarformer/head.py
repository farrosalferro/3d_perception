"""PolarFormer detection head in pure PyTorch (forward-only)."""

from __future__ import annotations

import copy
import math

import torch
from torch import nn

from .config import PolarFormerForwardConfig
from .postprocess import NMSFreeCoderLite
from .transformer import PolarTransformerLite
from .utils import SinePositionalEncoding2D, inverse_sigmoid


class PolarFormerHeadLite(nn.Module):
    """Standalone PolarFormer head that mirrors the reference forward contract."""

    def __init__(self, cfg: PolarFormerForwardConfig, transformer: PolarTransformerLite) -> None:
        super().__init__()
        self.cfg = cfg
        self.transformer = transformer
        self.embed_dims = cfg.embed_dims
        self.num_classes = cfg.num_classes
        self.num_query = cfg.num_queries
        self.code_size = cfg.code_size
        self.radius_range = cfg.polar_neck.radius_range

        self.input_projs = nn.ModuleList(
            [nn.Conv2d(cfg.backbone_neck.out_channels, self.embed_dims, kernel_size=1) for _ in range(cfg.polar_neck.num_levels)]
        )
        self.positional_encoding = SinePositionalEncoding2D(self.embed_dims // 2, normalize=True)
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
        self.cls_branches, self.reg_branches = self._build_branches(cfg.num_decoder_layers)

        self.bbox_coder = NMSFreeCoderLite(
            post_center_range=cfg.post_center_range,
            max_num=cfg.max_num,
            num_classes=cfg.num_classes,
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

    def forward(self, mlvl_feats: list[torch.Tensor], img_metas: list[dict]) -> dict[str, torch.Tensor]:
        if len(mlvl_feats) != len(self.input_projs):
            raise ValueError(
                f"Expected {len(self.input_projs)} feature levels, got {len(mlvl_feats)}."
            )
        _ = img_metas  # kept for parity with detector head API

        mlvl_proj_feats = []
        mlvl_masks = []
        mlvl_pos_embeds = []
        for feat, proj in zip(mlvl_feats, self.input_projs):
            projected = proj(feat)
            batch_size, _, height, width = projected.shape
            mask = torch.zeros((batch_size, height, width), dtype=torch.bool, device=projected.device)
            pos = self.positional_encoding(mask)
            mlvl_proj_feats.append(projected)
            mlvl_masks.append(mask)
            mlvl_pos_embeds.append(pos)

        query_ids = torch.arange(self.num_query, device=mlvl_proj_feats[0].device)
        query_embeds = self.query_embedding(query_ids)
        outs_dec, init_reference = self.transformer(
            mlvl_proj_feats,
            mlvl_masks,
            query_embeds,
            mlvl_pos_embeds,
        )
        outs_dec = torch.nan_to_num(outs_dec).permute(0, 2, 1, 3)  # [L, B, Q, C]
        reference = inverse_sigmoid(init_reference)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            tmp = self.reg_branches[lvl](outs_dec[lvl])

            tmp[..., 0:2] = (tmp[..., 0:2] + reference[..., 0:2]).sigmoid()
            tmp[..., 4:5] = (tmp[..., 4:5] + reference[..., 2:3]).sigmoid()

            theta = tmp[..., 0:1] * (2.0 * math.pi)
            radius = tmp[..., 1:2] * (self.radius_range[1] - self.radius_range[0]) + self.radius_range[0]
            z = tmp[..., 4:5] * (self.cfg.pc_range[5] - self.cfg.pc_range[2]) + self.cfg.pc_range[2]

            tmp_cart = tmp.clone()
            tmp_cart[..., 0:1] = torch.sin(theta) * radius
            tmp_cart[..., 1:2] = torch.cos(theta) * radius
            tmp_cart[..., 4:5] = z

            outputs_classes.append(outputs_class)
            outputs_coords.append(tmp_cart)

        return {
            "all_cls_scores": torch.stack(outputs_classes),
            "all_bbox_preds": torch.stack(outputs_coords),
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
        }

    def get_bboxes(self, preds_dicts: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        return self.bbox_coder.decode(preds_dicts)

