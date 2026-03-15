"""MapTR detection head in pure PyTorch (forward-only)."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from .config import MapTRForwardConfig
from .postprocess import MapTRNMSFreeCoderLite
from .transformer import MapTRPerceptionTransformerLite
from .utils import SinePositionalEncoding2D, inverse_sigmoid


class MapTRHeadLite(nn.Module):
    """Standalone MapTR head with MapTR-style vectorized polyline queries."""

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
        self.query_embed_type = cfg.query_embed_type
        self.with_box_refine = cfg.with_box_refine
        self.position_level = 0

        self.input_proj = nn.Conv2d(cfg.backbone_neck.out_channels, self.embed_dims, kernel_size=1)
        self.positional_encoding = SinePositionalEncoding2D(self.embed_dims // 2, normalize=True)
        self.adapt_pos3d = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1),
        )

        self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
        if self.query_embed_type == "instance_pts":
            self.instance_embedding = nn.Embedding(self.num_vec, self.embed_dims)
            self.pts_embedding = nn.Embedding(self.num_pts_per_vec, self.embed_dims)
            self.instance_content_embedding = nn.Embedding(self.num_vec, self.embed_dims)
            self.pts_content_embedding = nn.Embedding(self.num_pts_per_vec, self.embed_dims)
        elif self.query_embed_type == "all_pts":
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
            # Expose attributes for tests/hooks to keep API stable.
            self.instance_embedding = nn.Embedding(self.num_vec, self.embed_dims)
            self.pts_embedding = nn.Embedding(self.num_pts_per_vec, self.embed_dims)
            self.instance_content_embedding = nn.Embedding(self.num_vec, self.embed_dims)
            self.pts_content_embedding = nn.Embedding(self.num_pts_per_vec, self.embed_dims)
        else:
            raise ValueError(f"Unsupported query_embed_type: {self.query_embed_type}")
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

    @staticmethod
    def _parse_meta_hw(shape: object, *, key: str, batch_idx: int, cam_idx: int) -> tuple[int, int]:
        if not isinstance(shape, (list, tuple)) or len(shape) < 2:
            raise ValueError(
                f"img_metas[{batch_idx}]['{key}'][{cam_idx}] must contain at least (H, W), "
                f"got {shape!r}"
            )
        height, width = int(shape[0]), int(shape[1])
        if height <= 0 or width <= 0:
            raise ValueError(
                f"img_metas[{batch_idx}]['{key}'][{cam_idx}] must be positive, got {(height, width)}"
            )
        return height, width

    def _read_meta_hw_list(
        self,
        img_metas: list[dict],
        *,
        key: str,
        batch_idx: int,
        num_cams: int,
    ) -> list[tuple[int, int]]:
        if key not in img_metas[batch_idx]:
            raise KeyError(f"img_metas[{batch_idx}] must provide '{key}'.")
        values = img_metas[batch_idx][key]
        if not isinstance(values, (list, tuple)) or len(values) != num_cams:
            raise ValueError(
                f"img_metas[{batch_idx}]['{key}'] must have {num_cams} entries, got {values!r}"
            )
        return [
            self._parse_meta_hw(values[cam_idx], key=key, batch_idx=batch_idx, cam_idx=cam_idx)
            for cam_idx in range(num_cams)
        ]

    def _build_img_masks(
        self,
        img_metas: list[dict],
        *,
        batch_size: int,
        num_cams: int,
        device: torch.device,
    ) -> torch.Tensor:
        if len(img_metas) != batch_size:
            raise ValueError(f"img_metas length must be {batch_size}, got {len(img_metas)}")

        all_img_hw: list[list[tuple[int, int]]] = []
        max_pad_h, max_pad_w = 0, 0

        for batch_idx in range(batch_size):
            if not isinstance(img_metas[batch_idx], dict):
                raise TypeError(f"img_metas[{batch_idx}] must be a dict, got {type(img_metas[batch_idx])}")

            img_shapes = self._read_meta_hw_list(
                img_metas, key="img_shape", batch_idx=batch_idx, num_cams=num_cams
            )
            if "pad_shape" in img_metas[batch_idx] and img_metas[batch_idx]["pad_shape"] is not None:
                pad_shapes = self._read_meta_hw_list(
                    img_metas, key="pad_shape", batch_idx=batch_idx, num_cams=num_cams
                )
            elif self.cfg.strict_img_meta:
                raise KeyError("img_metas entries must provide 'pad_shape' when strict_img_meta=True.")
            else:
                pad_shapes = img_shapes

            for cam_idx in range(num_cams):
                img_h, img_w = img_shapes[cam_idx]
                pad_h, pad_w = pad_shapes[cam_idx]
                if pad_h < img_h or pad_w < img_w:
                    raise ValueError(
                        f"pad_shape must be >= img_shape for batch {batch_idx}, cam {cam_idx}; "
                        f"got pad={pad_shapes[cam_idx]}, img={img_shapes[cam_idx]}"
                    )
                max_pad_h = max(max_pad_h, pad_h)
                max_pad_w = max(max_pad_w, pad_w)

            all_img_hw.append(img_shapes)

        masks = torch.ones((batch_size, num_cams, max_pad_h, max_pad_w), device=device, dtype=torch.bool)
        for batch_idx in range(batch_size):
            for cam_idx in range(num_cams):
                img_h, img_w = all_img_hw[batch_idx][cam_idx]
                masks[batch_idx, cam_idx, :img_h, :img_w] = False
        return masks

    def _build_query_embedding(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.query_embed_type == "all_pts":
            return self.query_embedding.weight.to(device=device, dtype=dtype)

        instance_ids = torch.arange(self.num_vec, device=device)
        pts_ids = torch.arange(self.num_pts_per_vec, device=device)
        instance_pos = self.instance_embedding(instance_ids)
        pts_pos = self.pts_embedding(pts_ids)
        instance_content = self.instance_content_embedding(instance_ids)
        pts_content = self.pts_content_embedding(pts_ids)

        query_pos = instance_pos[:, None, :] + pts_pos[None, :, :]
        query_content = instance_content[:, None, :] + pts_content[None, :, :]
        object_query = torch.cat((query_pos, query_content), dim=-1)
        return object_query.reshape(self.num_query, self.embed_dims * 2).to(dtype=dtype)

    @staticmethod
    def _xyxy_to_cxcywh(boxes_xyxy: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        cx = 0.5 * (boxes_xyxy[..., 0] + boxes_xyxy[..., 2])
        cy = 0.5 * (boxes_xyxy[..., 1] + boxes_xyxy[..., 3])
        w = (boxes_xyxy[..., 2] - boxes_xyxy[..., 0]).clamp(min=eps)
        h = (boxes_xyxy[..., 3] - boxes_xyxy[..., 1]).clamp(min=eps)
        return torch.stack((cx, cy, w, h), dim=-1)

    def transform_box(self, pts: torch.Tensor, *, y_first: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert per-point query predictions to per-polyline boxes.

        Args:
            pts: [B, Q, code_size], where first 2 channels are point x/y logits.
            y_first: if True, channels are interpreted as [y, x].
        Returns:
            bbox_cxcywh: [B, V, 4]
            pts_xy: [B, V, P, 2]
        """
        if pts.dim() != 3:
            raise ValueError(f"pts must have shape [B, Q, C], got {tuple(pts.shape)}")
        if pts.shape[1] != self.num_query:
            raise ValueError(f"Expected Q={self.num_query}, got {pts.shape[1]}")
        if pts.shape[-1] < 2:
            raise ValueError(f"Expected at least 2 regression dims, got {pts.shape[-1]}")

        pts_xy = pts[..., :2].view(pts.shape[0], self.num_vec, self.num_pts_per_vec, 2)
        pts_y = pts_xy[..., 0] if y_first else pts_xy[..., 1]
        pts_x = pts_xy[..., 1] if y_first else pts_xy[..., 0]

        xmin = pts_x.min(dim=2, keepdim=True).values
        xmax = pts_x.max(dim=2, keepdim=True).values
        ymin = pts_y.min(dim=2, keepdim=True).values
        ymax = pts_y.max(dim=2, keepdim=True).values
        bbox_xyxy = torch.cat((xmin, ymin, xmax, ymax), dim=2)
        bbox_cxcywh = self._xyxy_to_cxcywh(bbox_xyxy)
        return bbox_cxcywh, pts_xy

    def forward(self, mlvl_feats: list[torch.Tensor], img_metas: list[dict]) -> dict[str, torch.Tensor]:
        x = mlvl_feats[self.position_level]
        batch_size, num_cams = x.shape[:2]
        if self.cfg.strict_img_meta and num_cams != self.cfg.num_cams:
            raise ValueError(f"Expected num_cams={self.cfg.num_cams}, got {num_cams}")

        masks = self._build_img_masks(img_metas, batch_size=batch_size, num_cams=num_cams, device=x.device)

        x = self.input_proj(x.flatten(0, 1)).view(batch_size, num_cams, self.embed_dims, x.shape[-2], x.shape[-1])
        masks = F.interpolate(masks.float(), size=x.shape[-2:], mode="nearest").to(torch.bool)

        pos_embeds = [self.positional_encoding(masks[:, cam_idx]).unsqueeze(1) for cam_idx in range(num_cams)]
        cam_pos_embed = torch.cat(pos_embeds, dim=1)
        cam_pos_embed = self.adapt_pos3d(cam_pos_embed.flatten(0, 1)).view_as(x)

        query_embed = self._build_query_embedding(device=x.device, dtype=x.dtype)
        bev_ids = torch.arange(self.bev_h * self.bev_w, device=x.device)
        bev_queries = self.bev_embedding(bev_ids).to(dtype=x.dtype)
        bev_mask = torch.zeros((batch_size, self.bev_h, self.bev_w), device=x.device, dtype=torch.bool)
        bev_pos = self.positional_encoding(bev_mask).flatten(2).transpose(1, 2).to(dtype=x.dtype)

        bev_embed, hs, init_reference, inter_references = self.transformer(
            x,
            masks,
            query_embed,
            cam_pos_embed,
            bev_queries,
            bev_pos,
            reg_branches=self.reg_branches if self.with_box_refine else None,
        )
        hs = torch.nan_to_num(hs).permute(0, 2, 1, 3).contiguous()  # [L, B, Q, C]
        init_reference = torch.nan_to_num(init_reference)
        inter_references = torch.nan_to_num(inter_references)

        outputs_classes = []
        outputs_coords = []
        outputs_pts_coords = []
        for lvl in range(hs.shape[0]):
            hidden = hs[lvl]
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            vec_embedding = hidden.reshape(batch_size, self.num_vec, self.num_pts_per_vec, -1).mean(2)
            outputs_class = self.cls_branches[lvl](vec_embedding)
            tmp = self.reg_branches[lvl](hidden)
            tmp = tmp.clone()
            tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2]
            tmp = tmp.sigmoid()
            outputs_coord, outputs_pts_coord = self.transform_box(tmp)

            outputs_classes.append(outputs_class)
            outputs_pts_coords.append(outputs_pts_coord)
            outputs_coords.append(outputs_coord)

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
