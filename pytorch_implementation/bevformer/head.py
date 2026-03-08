"""BEVFormer detection head in pure PyTorch (forward-only)."""

from __future__ import annotations

import copy

import torch
from torch import nn

from .config import BEVFormerForwardConfig
from .modules import PerceptionTransformerLite
from .postprocess import NMSFreeCoderLite
from .utils.math import inverse_sigmoid
from .utils.positional_encoding import LearnedPositionalEncoding2D


class BEVFormerHeadLite(nn.Module):
    """Standalone BEVFormer head for forward-only experimentation."""

    def __init__(
        self,
        cfg: BEVFormerForwardConfig,
        transformer: PerceptionTransformerLite,
        *,
        with_box_refine: bool = True,
        as_two_stage: bool = False,
        code_size: int = 10,
        num_reg_fcs: int = 2,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.transformer = transformer
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.code_size = code_size
        self.num_reg_fcs = num_reg_fcs
        self.embed_dims = cfg.embed_dims
        self.bev_h = cfg.bev_h
        self.bev_w = cfg.bev_w
        self.num_queries = cfg.num_queries
        self.num_classes = cfg.num_classes
        self.pc_range = cfg.pc_range
        self.real_h = cfg.real_h
        self.real_w = cfg.real_w

        self.positional_encoding = LearnedPositionalEncoding2D(
            num_feats=cfg.embed_dims // 2,
            row_num_embed=self.bev_h,
            col_num_embed=self.bev_w,
        )
        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims * 2)
        self.cls_branches, self.reg_branches = self._build_branches()

        self.bbox_coder = NMSFreeCoderLite(
            pc_range=cfg.pc_range,
            post_center_range=cfg.post_center_range,
            max_num=cfg.max_num,
            score_threshold=cfg.score_threshold,
            num_classes=cfg.num_classes,
        )

    def _build_branches(self) -> tuple[nn.ModuleList, nn.ModuleList]:
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.num_classes))
        cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU(inplace=True))
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        num_pred = self.transformer.decoder.layers.__len__()
        if self.with_box_refine:
            cls_branches = nn.ModuleList([copy.deepcopy(cls_branch) for _ in range(num_pred)])
            reg_branches = nn.ModuleList([copy.deepcopy(reg_branch) for _ in range(num_pred)])
        else:
            cls_branches = nn.ModuleList([cls_branch for _ in range(num_pred)])
            reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])
        return cls_branches, reg_branches

    def forward(
        self,
        mlvl_feats: list[torch.Tensor],
        img_metas: list[dict],
        *,
        prev_bev: torch.Tensor | None = None,
        only_bev: bool = False,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        bs, _, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        bev_queries = self.bev_embedding.weight.to(dtype)
        object_query_embeds = self.query_embedding.weight.to(dtype)
        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=bev_queries.device, dtype=dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        if only_bev:
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )

        bev_embed, hs, init_reference, inter_references = self.transformer(
            mlvl_feats,
            bev_queries,
            object_query_embeds,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            cls_branches=None,
            img_metas=img_metas,
            prev_bev=prev_bev,
        )

        hs = hs.permute(0, 2, 1, 3)  # [num_dec, bs, num_query, embed]
        outputs_classes = []
        outputs_coords = []
        for level_idx in range(hs.shape[0]):
            reference = init_reference if level_idx == 0 else inter_references[level_idx - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[level_idx](hs[level_idx])
            tmp = self.reg_branches[level_idx](hs[level_idx])
            if reference.shape[-1] != 3:
                raise ValueError("reference must have last dimension=3.")

            tmp[..., 0:2] = (tmp[..., 0:2] + reference[..., 0:2]).sigmoid()
            tmp[..., 4:5] = (tmp[..., 4:5] + reference[..., 2:3]).sigmoid()
            tmp[..., 0:1] = tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            tmp[..., 1:2] = tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            tmp[..., 4:5] = tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            outputs_classes.append(outputs_class)
            outputs_coords.append(tmp)

        return {
            "bev_embed": bev_embed,
            "all_cls_scores": torch.stack(outputs_classes),
            "all_bbox_preds": torch.stack(outputs_coords),
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
        }

    def get_bboxes(self, preds_dicts: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        decoded = self.bbox_coder.decode(preds_dicts)
        results = []
        for sample in decoded:
            bboxes = sample["bboxes"].clone()
            if bboxes.numel() > 0 and bboxes.shape[-1] >= 6:
                bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            results.append(
                {
                    "bboxes": bboxes,
                    "scores": sample["scores"],
                    "labels": sample["labels"],
                }
            )
        return results
