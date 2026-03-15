"""Core Sparse4D blocks implemented in pure PyTorch."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ = list(range(11))


class SparseBox3DEncoderLite(nn.Module):
    """Encodes raw anchor states into query-aligned positional embeddings."""

    def __init__(self, box_code_size: int, embed_dims: int) -> None:
        super().__init__()
        hidden = max(embed_dims // 2, 64)
        self.mlp = nn.Sequential(
            nn.Linear(box_code_size, hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, embed_dims),
        )

    def forward(self, anchors: torch.Tensor) -> torch.Tensor:
        return self.mlp(anchors)


class SparseBox3DKeyPointsGeneratorLite(nn.Module):
    """Generates anchor-centric 3D key points used for feature sampling."""

    def __init__(
        self,
        box_code_size: int,
        *,
        fix_scale: Sequence[Sequence[float]] | None = None,
    ) -> None:
        super().__init__()
        if fix_scale is None:
            fix_scale = (
                (0.0, 0.0, 0.0),
                (0.45, 0.0, 0.0),
                (-0.45, 0.0, 0.0),
                (0.0, 0.45, 0.0),
                (0.0, -0.45, 0.0),
                (0.0, 0.0, 0.45),
                (0.0, 0.0, -0.45),
            )
        self.box_code_size = int(box_code_size)
        self.register_buffer("fix_scale", torch.tensor(fix_scale, dtype=torch.float32))
        self.num_pts = int(self.fix_scale.shape[0])

    def forward(self, anchor: torch.Tensor) -> torch.Tensor:
        batch_size, num_anchor = anchor.shape[:2]
        if self.box_code_size >= 6:
            size = anchor[..., [W, L, H]].exp().unsqueeze(-2)
        else:
            size = anchor.new_ones(batch_size, num_anchor, 1, 3)
        key_points = self.fix_scale.to(device=anchor.device, dtype=anchor.dtype) * size

        if self.box_code_size >= 8:
            rotation = anchor.new_zeros(batch_size, num_anchor, 3, 3)
            rotation[..., 0, 0] = anchor[..., COS_YAW]
            rotation[..., 0, 1] = -anchor[..., SIN_YAW]
            rotation[..., 1, 0] = anchor[..., SIN_YAW]
            rotation[..., 1, 1] = anchor[..., COS_YAW]
            rotation[..., 2, 2] = 1.0
            key_points = torch.matmul(rotation[:, :, None], key_points.unsqueeze(-1)).squeeze(-1)

        if self.box_code_size >= 3:
            key_points = key_points + anchor[..., None, [X, Y, Z]]
        return key_points

    @staticmethod
    def anchor_distance(anchor: torch.Tensor) -> torch.Tensor:
        return torch.norm(anchor[..., :2], p=2, dim=-1)


class DeformableFeatureAggregationLite(nn.Module):
    """Projection-aware multi-view feature sampling and fusion."""

    def __init__(
        self,
        *,
        embed_dims: int,
        box_code_size: int,
        num_groups: int,
        num_levels: int,
        num_cams: int,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims ({embed_dims}) must be divisible by num_groups ({num_groups})."
            )
        self.embed_dims = int(embed_dims)
        self.num_groups = int(num_groups)
        self.group_dims = self.embed_dims // self.num_groups
        self.num_levels = int(num_levels)
        self.num_cams = int(num_cams)
        self.attn_drop = float(attn_drop)

        self.kps_generator = SparseBox3DKeyPointsGeneratorLite(box_code_size=box_code_size)
        self.num_pts = self.kps_generator.num_pts
        self.weights_fc = nn.Linear(
            self.embed_dims,
            self.num_groups * self.num_cams * self.num_levels * self.num_pts,
        )
        self.output_proj = nn.Linear(self.embed_dims, self.embed_dims)

    @staticmethod
    def project_points(
        key_points: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project 3D key points to normalized image coordinates."""

        batch_size = key_points.shape[0]
        if projection_mat.dim() != 4 or projection_mat.shape[0] != batch_size:
            raise ValueError(
                "projection_mat must have shape [B, Ncam, 3/4, 4] and match key point batch size."
            )
        if projection_mat.shape[-2:] not in ((3, 4), (4, 4)):
            raise ValueError("projection_mat must have trailing shape [3, 4] or [4, 4].")

        pts_extend = torch.cat([key_points, torch.ones_like(key_points[..., :1])], dim=-1)
        points_2d = torch.matmul(
            projection_mat[:, :, None, None],
            pts_extend[:, None, ..., None],
        ).squeeze(-1)
        if points_2d.shape[-1] == 4:
            points_2d = points_2d[..., :3]
        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)

        if image_wh is not None:
            if image_wh.shape[:2] != projection_mat.shape[:2] or image_wh.shape[-1] != 2:
                raise ValueError("image_wh must have shape [B, Ncam, 2].")
            points_2d = points_2d / torch.clamp(image_wh[:, :, None, None], min=1e-5)
        return points_2d

    @staticmethod
    def feature_sampling(
        feature_maps: Sequence[torch.Tensor],
        key_points: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample multilevel image features at projected key points."""

        if not feature_maps:
            raise ValueError("feature_maps must include at least one level.")
        num_levels = len(feature_maps)
        batch_size, num_anchor, num_pts = key_points.shape[:3]
        num_cams = feature_maps[0].shape[1]

        points_2d = DeformableFeatureAggregationLite.project_points(
            key_points,
            projection_mat,
            image_wh=image_wh,
        )
        points_2d = points_2d * 2.0 - 1.0
        points_2d = points_2d.flatten(end_dim=1)

        sampled_levels = []
        for level_feature in feature_maps:
            if level_feature.dim() != 5:
                raise ValueError("Each feature level must have shape [B, Ncam, C, H, W].")
            if level_feature.shape[0] != batch_size or level_feature.shape[1] != num_cams:
                raise ValueError("Feature level batch/camera dimensions must match key points.")
            sampled = F.grid_sample(
                level_feature.flatten(end_dim=1),
                points_2d,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled_levels.append(sampled)

        features = torch.stack(sampled_levels, dim=1)
        features = features.reshape(
            batch_size,
            num_cams,
            num_levels,
            -1,
            num_anchor,
            num_pts,
        ).permute(0, 4, 1, 2, 5, 3)
        return features

    def _get_weights(
        self,
        instance_feature: torch.Tensor,
        anchor_embed: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed
        weights = (
            self.weights_fc(feature)
            .reshape(batch_size, num_anchor, -1, self.num_groups)
            .softmax(dim=-2)
            .reshape(
                batch_size,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )
        if self.training and self.attn_drop > 0:
            mask = torch.rand(
                batch_size,
                num_anchor,
                self.num_cams,
                1,
                self.num_pts,
                1,
                device=weights.device,
                dtype=weights.dtype,
            )
            weights = ((mask > self.attn_drop) * weights) / (1.0 - self.attn_drop)
        return weights

    def multi_view_level_fusion(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_anchor = weights.shape[:2]
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )
        features = features.sum(dim=2).sum(dim=2)
        features = features.reshape(batch_size, num_anchor, self.num_pts, self.embed_dims)
        return features

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        img_feats: Sequence[torch.Tensor],
        projection_mat: torch.Tensor | None = None,
        image_wh: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not img_feats:
            raise ValueError("img_feats must contain at least one feature level.")
        if projection_mat is None:
            raise ValueError("projection_mat is required for projection-aware feature sampling.")
        if len(img_feats) != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} feature levels, but got {len(img_feats)}."
            )
        if img_feats[0].shape[1] != self.num_cams:
            raise ValueError(
                f"Expected {self.num_cams} cameras, but got {img_feats[0].shape[1]}."
            )

        key_points = self.kps_generator(anchor)
        weights = self._get_weights(instance_feature, anchor_embed)
        features = self.feature_sampling(
            img_feats,
            key_points,
            projection_mat,
            image_wh=image_wh,
        )
        features = self.multi_view_level_fusion(features, weights)
        features = features.sum(dim=2)
        return self.output_proj(features)


class SparseDecoderLayerLite(nn.Module):
    """Sparse4D decoder layer: self-attn + temporal-attn + deformable fusion + FFN."""

    def __init__(
        self,
        *,
        embed_dims: int,
        num_heads: int,
        ffn_dims: int,
        dropout: float,
        box_code_size: int,
        num_levels: int,
        num_cams: int,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temp_attn = nn.MultiheadAttention(
            embed_dim=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = DeformableFeatureAggregationLite(
            embed_dims=embed_dims,
            box_code_size=box_code_size,
            num_groups=num_heads,
            num_levels=num_levels,
            num_cams=num_cams,
            attn_drop=dropout,
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, ffn_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dims, embed_dims),
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm_temp = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

    def forward(
        self,
        query: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        img_feats: Sequence[torch.Tensor],
        projection_mat: torch.Tensor | None = None,
        image_wh: torch.Tensor | None = None,
        temp_instance_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self_out = self.self_attn(query, query, query, need_weights=False)[0]
        query = self.norm1(query + self_out)

        if temp_instance_feature is not None:
            temp_out = self.temp_attn(
                query,
                temp_instance_feature,
                temp_instance_feature,
                need_weights=False,
            )[0]
            query = self.norm_temp(query + temp_out)

        cross_out = self.cross_attn(
            query,
            anchor,
            anchor_embed,
            img_feats,
            projection_mat=projection_mat,
            image_wh=image_wh,
        )
        query = self.norm2(query + cross_out)
        ffn_out = self.ffn(query)
        query = self.norm3(query + ffn_out)
        return query


class SparseBox3DRefinementLite(nn.Module):
    """Predicts class logits and refined anchor states per decoder layer."""

    def __init__(
        self,
        embed_dims: int,
        num_classes: int,
        box_code_size: int,
        *,
        normalize_yaw: bool = False,
        refine_yaw: bool = True,
    ) -> None:
        super().__init__()
        self.box_code_size = int(box_code_size)
        self.normalize_yaw = bool(normalize_yaw)
        self.refine_yaw = bool(refine_yaw)
        self.refine_state = [X, Y, Z, W, L, H]
        if self.refine_yaw and self.box_code_size >= 8:
            self.refine_state += [SIN_YAW, COS_YAW]

        self.cls_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, num_classes),
        )
        self.reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, box_code_size),
        )

    def forward(
        self,
        query: torch.Tensor,
        anchors: torch.Tensor,
        anchor_embed: torch.Tensor,
        *,
        time_interval: torch.Tensor | float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature = query + anchor_embed
        refined = self.reg_branch(feature)
        refined[..., self.refine_state] = (
            refined[..., self.refine_state] + anchors[..., self.refine_state]
        )

        if self.normalize_yaw and self.box_code_size >= 8:
            refined[..., [SIN_YAW, COS_YAW]] = F.normalize(
                refined[..., [SIN_YAW, COS_YAW]],
                dim=-1,
            )

        if self.box_code_size > VX:
            if not torch.is_tensor(time_interval):
                interval = refined.new_tensor(time_interval)
            else:
                interval = time_interval.to(device=refined.device, dtype=refined.dtype)
            interval = torch.clamp(interval.reshape(-1, 1, 1), min=1e-3)
            translation = refined[..., VX:]
            velocity = translation / interval
            refined[..., VX:] = velocity + anchors[..., VX:]

        cls_scores = self.cls_branch(query)
        return cls_scores, refined
