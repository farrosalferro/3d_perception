"""FB-BEV detection head in pure PyTorch (forward-only)."""

from __future__ import annotations

import torch
from torch import nn

from .config import FBBEVForwardConfig
from .postprocess import FBBEVBoxCoderLite


class FBBEVDetectionHeadLite(nn.Module):
    """Center-based detection head with query-style outputs."""

    def __init__(self, cfg: FBBEVForwardConfig) -> None:
        super().__init__()
        self.cfg = cfg
        channels = cfg.embed_dims
        self.num_classes = cfg.num_classes
        self.code_size = cfg.code_size
        self.max_num = cfg.max_num
        self.occupancy_classes = int(cfg.occupancy_classes)
        self.fix_void = bool(cfg.occupancy_fix_void)
        self.pc_range = cfg.pc_range
        self.bev_h = cfg.bev_h
        self.bev_w = cfg.bev_w

        self.shared = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.heatmap_head = nn.Conv2d(channels, self.num_classes, kernel_size=1)
        self.reg_head = nn.Conv2d(channels, self.code_size, kernel_size=1)
        occ_channels = max(channels // 2, 1)
        self.occ_refine = nn.Sequential(
            nn.Conv3d(channels, occ_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(occ_channels),
            nn.ReLU(inplace=True),
        )
        self.occ_classifier = nn.Conv3d(occ_channels, self.occupancy_classes, kernel_size=1)
        self.bbox_coder = FBBEVBoxCoderLite(
            post_center_range=cfg.post_center_range,
            max_num=cfg.max_num,
            num_classes=cfg.num_classes,
            score_threshold=cfg.score_threshold,
        )

    def _meshgrid(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        y = torch.arange(self.bev_h, device=device, dtype=dtype)
        x = torch.arange(self.bev_w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return grid_x, grid_y

    def _decode_reg_map(self, reg_map: torch.Tensor) -> torch.Tensor:
        """Convert dense map to metric-space boxes.

        reg_map layout per location:
            [dx, dy, z, log_w, log_l, log_h, yaw, vx, vy]
        """

        batch_size, _, height, width = reg_map.shape
        grid_x, grid_y = self._meshgrid(reg_map.device, reg_map.dtype)
        grid_x = grid_x.view(1, 1, height, width).expand(batch_size, 1, height, width)
        grid_y = grid_y.view(1, 1, height, width).expand(batch_size, 1, height, width)

        x = (grid_x + reg_map[:, 0:1].sigmoid()) / max(width - 1, 1)
        y = (grid_y + reg_map[:, 1:2].sigmoid()) / max(height - 1, 1)
        x = x * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        y = y * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        z = reg_map[:, 2:3]
        dims = reg_map[:, 3:6].exp()
        yaw = reg_map[:, 6:7]
        vel = reg_map[:, 7:9]
        return torch.cat([x, y, z, dims, yaw, vel], dim=1)

    def _select_topk_queries(self, cls_map: torch.Tensor, bbox_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, height, width = cls_map.shape
        logits_flat = cls_map.permute(0, 2, 3, 1).reshape(batch_size, height * width, self.num_classes)
        scores = logits_flat.sigmoid().amax(dim=-1)
        topk = min(self.max_num, int(scores.shape[1]))
        topk_scores, topk_indices = scores.topk(topk, dim=1)

        gather_idx_cls = topk_indices.unsqueeze(-1).expand(-1, -1, self.num_classes)
        query_cls = torch.gather(logits_flat, 1, gather_idx_cls)

        bbox_flat = bbox_map.permute(0, 2, 3, 1).reshape(batch_size, height * width, self.code_size)
        gather_idx_box = topk_indices.unsqueeze(-1).expand(-1, -1, self.code_size)
        query_bbox = torch.gather(bbox_flat, 1, gather_idx_box)

        # Preserve confidence in logits tensor by adding top-k score residual.
        query_cls = query_cls + topk_scores.unsqueeze(-1).detach() * 0.0
        return query_cls, query_bbox

    def _forward_occupancy(self, bev_volume: torch.Tensor) -> torch.Tensor:
        if bev_volume.dim() != 5:
            raise ValueError(f"Expected bev_volume [B, C, H, W, Z], got {tuple(bev_volume.shape)}")
        occ_input = bev_volume.permute(0, 1, 4, 2, 3).contiguous()  # [B, C, Z, H, W]
        occ_feat = self.occ_refine(occ_input)
        occ_logits = self.occ_classifier(occ_feat).permute(0, 1, 3, 4, 2).contiguous()
        return occ_logits

    def forward(self, bev_embed: torch.Tensor, *, bev_volume: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if bev_embed.dim() != 4:
            raise ValueError(f"Expected bev_embed [B, C, H, W], got {tuple(bev_embed.shape)}")
        shared = self.shared(bev_embed)
        cls_map = self.heatmap_head(shared)
        reg_map = self.reg_head(shared)
        bbox_map = self._decode_reg_map(reg_map)
        query_cls, query_bbox = self._select_topk_queries(cls_map, bbox_map)
        if bev_volume is None:
            bev_volume = bev_embed.unsqueeze(-1)
        occupancy_logits = self._forward_occupancy(bev_volume)
        return {
            "all_cls_scores": query_cls.unsqueeze(0),
            "all_bbox_preds": query_bbox.unsqueeze(0),
            "dense_cls_logits": cls_map,
            "dense_bbox_map": bbox_map,
            "occupancy_logits": occupancy_logits,
            "output_voxels": [occupancy_logits],
        }

    def get_bboxes(self, preds_dicts: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        return self.bbox_coder.decode(preds_dicts)

    def decode_occupancy(
        self,
        preds_dicts: dict[str, torch.Tensor],
        *,
        fix_void: bool | None = None,
        return_raw_occ: bool = False,
    ) -> list[torch.Tensor]:
        occupancy_logits = preds_dicts["occupancy_logits"]
        return self.bbox_coder.decode_occupancy(
            occupancy_logits,
            fix_void=self.fix_void if fix_void is None else bool(fix_void),
            return_raw_occ=return_raw_occ,
        )
