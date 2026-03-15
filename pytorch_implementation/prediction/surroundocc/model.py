"""Pure-PyTorch prediction/surroundocc forward path with strict parity semantics."""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ..common.time_contracts import (
    resolve_time_indices as _shared_resolve_time_indices,
    time_deltas_from_indices as _shared_time_deltas_from_indices,
)
from .config import SurroundOccPredictionConfig
from .postprocess import decode_predictions


def _conv_norm_act(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.GELU(),
    )


def _coerce_time_indices(
    time_indices: torch.Tensor | Sequence[float] | None,
    *,
    expected_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    return _shared_resolve_time_indices(
        time_indices,
        expected_steps=expected_steps,
        device=device,
        dtype=dtype,
        name=name,
        require_strictly_increasing=True,
    )


def _time_deltas_from_indices(time_indices: torch.Tensor) -> torch.Tensor:
    return _shared_time_deltas_from_indices(time_indices)


def _coerce_hw_pairs(
    value: Any,
    *,
    num_cams: int,
    field_name: str,
    batch_index: int,
) -> list[tuple[int, int]]:
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()

    pairs: list[tuple[int, int]]
    if isinstance(value, (list, tuple)) and len(value) >= 2 and isinstance(value[0], (int, float)):
        h, w = int(value[0]), int(value[1])
        pairs = [(h, w)] * num_cams
    elif isinstance(value, (list, tuple)) and len(value) == num_cams:
        pairs = []
        for cam_idx, item in enumerate(value):
            if torch.is_tensor(item):
                item = item.detach().cpu().tolist()
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                raise ValueError(
                    f"{field_name} for batch {batch_index}, cam {cam_idx} must provide (h, w), got {item}."
                )
            pairs.append((int(item[0]), int(item[1])))
    else:
        raise ValueError(
            f"{field_name} for batch {batch_index} must be (h, w) or per-camera list with {num_cams} entries."
        )

    for cam_idx, (h, w) in enumerate(pairs):
        if h <= 0 or w <= 0:
            raise ValueError(
                f"{field_name} for batch {batch_index}, cam {cam_idx} must have positive sizes, got {(h, w)}."
            )
    return pairs


def _validate_and_stack_metas(
    img_metas: list[dict[str, Any]],
    *,
    batch_size: int,
    num_cams: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(img_metas, list) or len(img_metas) != batch_size:
        raise ValueError(f"img_metas must be a list with length={batch_size}.")

    required_keys = {"lidar2img", "img_shape", "pad_shape"}
    lidar2img_batch: list[torch.Tensor] = []
    img_hw_batch: list[torch.Tensor] = []

    for batch_idx, meta in enumerate(img_metas):
        if not isinstance(meta, dict):
            raise ValueError(f"img_metas[{batch_idx}] must be a dict, got {type(meta).__name__}.")
        missing = required_keys - set(meta.keys())
        if missing:
            raise ValueError(f"img_metas[{batch_idx}] is missing required keys: {sorted(missing)}.")

        lidar2img = torch.as_tensor(meta["lidar2img"], device=device, dtype=dtype)
        if lidar2img.shape != (num_cams, 4, 4):
            raise ValueError(
                f"img_metas[{batch_idx}]['lidar2img'] must have shape {(num_cams, 4, 4)}, "
                f"got {tuple(lidar2img.shape)}."
            )
        if not torch.isfinite(lidar2img).all():
            raise ValueError(f"img_metas[{batch_idx}]['lidar2img'] must be finite.")
        lidar2img_batch.append(lidar2img)

        img_pairs = _coerce_hw_pairs(meta["img_shape"], num_cams=num_cams, field_name="img_shape", batch_index=batch_idx)
        pad_pairs = _coerce_hw_pairs(meta["pad_shape"], num_cams=num_cams, field_name="pad_shape", batch_index=batch_idx)
        for cam_idx, ((img_h, img_w), (pad_h, pad_w)) in enumerate(zip(img_pairs, pad_pairs)):
            if pad_h < img_h or pad_w < img_w:
                raise ValueError(
                    f"pad_shape must be >= img_shape for batch {batch_idx}, cam {cam_idx}; "
                    f"got pad={(pad_h, pad_w)} img={(img_h, img_w)}."
                )
        img_hw_batch.append(torch.tensor(img_pairs, device=device, dtype=dtype))

    return torch.stack(lidar2img_batch, dim=0), torch.stack(img_hw_batch, dim=0)


def _as_camera_feature_levels(
    camera_features: torch.Tensor | Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    if torch.is_tensor(camera_features):
        return [camera_features]
    if isinstance(camera_features, Sequence):
        levels = list(camera_features)
        if not levels:
            raise ValueError("camera_features must include at least one feature level.")
        if not all(torch.is_tensor(level) for level in levels):
            raise TypeError("camera_features levels must be tensors with shape [B, Ncam, C, H, W].")
        return levels
    raise TypeError("camera_features must be a tensor or a sequence of tensors.")


class SpatialEncoder(nn.Module):
    """Encode BEV history into compact spatial memory."""

    def __init__(self, in_channels: int, embed_dims: int) -> None:
        super().__init__()
        mid_channels = max(embed_dims // 2, 16)
        self.stem = _conv_norm_act(in_channels, mid_channels, stride=1)
        self.block1 = _conv_norm_act(mid_channels, mid_channels, stride=2)
        self.block2 = _conv_norm_act(mid_channels, embed_dims, stride=2)
        self.out_proj = _conv_norm_act(embed_dims, embed_dims, stride=1)

    def forward(self, history_bev: torch.Tensor) -> torch.Tensor:
        if history_bev.dim() != 5:
            raise ValueError(
                "Expected history_bev shape [B, T, C, Hb, Wb], "
                f"got {tuple(history_bev.shape)}."
            )
        batch_size, history_steps, channels, height, width = history_bev.shape
        x = history_bev.reshape(batch_size * history_steps, channels, height, width)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.out_proj(x)
        return x.view(batch_size, history_steps, x.shape[1], x.shape[2], x.shape[3])


class TemporalContextEncoder(nn.Module):
    """Encode temporal history with a recurrent context."""

    def __init__(self, embed_dims: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=embed_dims,
            hidden_size=embed_dims,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.norm = nn.LayerNorm(embed_dims)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        spatial_features: torch.Tensor,
        history_time_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, history_steps = spatial_features.shape[:2]
        temporal_tokens = spatial_features.mean(dim=(-1, -2))  # [B, T, C]
        if history_time_indices is not None:
            if history_time_indices.dim() == 1:
                time_deltas = _time_deltas_from_indices(history_time_indices).unsqueeze(0).expand(batch_size, -1)
            elif history_time_indices.dim() == 2:
                if history_time_indices.shape[0] != batch_size:
                    raise ValueError(
                        "history_time_indices batch size must match spatial features. "
                        f"Got {history_time_indices.shape[0]} vs {batch_size}."
                    )
                time_deltas = _time_deltas_from_indices(history_time_indices)
            else:
                raise ValueError(
                    f"history_time_indices must be 1D or 2D, got {tuple(history_time_indices.shape)}."
                )
            if time_deltas.shape[-1] != history_steps:
                raise ValueError(
                    f"history_time_indices must have history length={history_steps}, "
                    f"got {time_deltas.shape[-1]}."
                )
            time_scale = time_deltas / time_deltas.mean(dim=-1, keepdim=True).clamp(min=1e-6)
            temporal_tokens = temporal_tokens * time_scale.unsqueeze(-1)

        sequence, _ = self.gru(temporal_tokens)
        sequence = self.dropout(self.norm(sequence))
        context = sequence[:, -1]
        return sequence, context


class HorizonDecoder(nn.Module):
    """Project temporal context into horizon-specific BEV features."""

    def __init__(self, embed_dims: int, future_steps: int, dropout: float) -> None:
        super().__init__()
        self.future_steps = int(future_steps)
        self.horizon_embedding = nn.Embedding(self.future_steps, embed_dims)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, embed_dims),
            nn.GELU(),
            nn.Linear(embed_dims, embed_dims),
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dims, embed_dims),
        )

    def forward(
        self,
        context: torch.Tensor,
        last_spatial: torch.Tensor,
        future_time_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = context.shape[0]
        if future_time_indices.dim() == 1:
            time_embed = self.time_mlp(future_time_indices.unsqueeze(-1))  # [H, C]
            horizon_index = torch.arange(self.future_steps, device=context.device)
            horizon_base = self.horizon_embedding(horizon_index) + time_embed
            horizon_tokens = context[:, None, :] + horizon_base[None, :, :]
        elif future_time_indices.dim() == 2:
            if future_time_indices.shape[0] != batch_size:
                raise ValueError(
                    "future_time_indices batch size must match context batch size. "
                    f"Got {future_time_indices.shape[0]} vs {batch_size}."
                )
            time_embed = self.time_mlp(future_time_indices.unsqueeze(-1))  # [B, H, C]
            horizon_index = torch.arange(self.future_steps, device=context.device)
            horizon_base = self.horizon_embedding(horizon_index)[None, :, :] + time_embed
            horizon_tokens = context[:, None, :] + horizon_base
        else:
            raise ValueError(
                f"future_time_indices must be 1D or 2D, got {tuple(future_time_indices.shape)}."
            )
        horizon_tokens = self.mlp(horizon_tokens)

        horizon_features = last_spatial[:, None, ...] + horizon_tokens[:, :, :, None, None]
        return horizon_features, horizon_tokens


class CameraLevelProjector(nn.Module):
    """Project each camera level to the shared transformer embedding dimension."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.projections = nn.ModuleDict()

    def _get_or_create_projection(
        self,
        in_channels: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Conv2d:
        key = str(int(in_channels))
        if key not in self.projections:
            projection = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, bias=False)
            nn.init.xavier_uniform_(projection.weight)
            self.projections[key] = projection
        projection = self.projections[key]
        if projection.weight.device != device or projection.weight.dtype != dtype:
            projection = projection.to(device=device, dtype=dtype)
            self.projections[key] = projection
        return projection

    def forward(self, level_features: torch.Tensor) -> torch.Tensor:
        if level_features.dim() != 5:
            raise ValueError(
                "Each camera feature level must have shape [B, Ncam, C, H, W], "
                f"got {tuple(level_features.shape)}."
            )
        batch_size, num_cams, channels, height, width = level_features.shape
        projection = self._get_or_create_projection(
            channels,
            device=level_features.device,
            dtype=level_features.dtype,
        )
        flattened = level_features.reshape(batch_size * num_cams, channels, height, width)
        projected = projection(flattened)
        return projected.view(batch_size, num_cams, self.out_channels, height, width)


class CameraViewFusion(nn.Module):
    """Pure-PyTorch camera-view fusion that mirrors upstream voxel query semantics."""

    def __init__(self, cfg: SurroundOccPredictionConfig) -> None:
        super().__init__()
        self.num_cams = int(cfg.num_cams)
        self.depth_bins = int(cfg.depth_bins)
        self.embed_dims = int(cfg.embed_dims)
        self.pc_range = tuple(float(v) for v in cfg.pc_range)

        self.level_projector = CameraLevelProjector(self.embed_dims)
        self.query_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.output_norm = nn.LayerNorm(self.embed_dims)
        self.dropout = nn.Dropout(cfg.dropout)

    @staticmethod
    def _build_reference_points(
        batch_size: int,
        *,
        height: int,
        width: int,
        depth: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        zs = torch.linspace(0.5, depth - 0.5, depth, device=device, dtype=dtype) / float(depth)
        ys = torch.linspace(0.5, height - 0.5, height, device=device, dtype=dtype) / float(height)
        xs = torch.linspace(0.5, width - 0.5, width, device=device, dtype=dtype) / float(width)
        try:
            zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
        except TypeError:  # pragma: no cover - compatibility for older torch
            zz, yy, xx = torch.meshgrid(zs, ys, xs)
        points = torch.stack((xx, yy, zz), dim=-1)  # [D, H, W, 3]
        points = points.permute(1, 2, 0, 3).reshape(1, height * width, depth, 3)
        return points.repeat(batch_size, 1, 1, 1)

    def _point_sampling(
        self,
        reference_points: torch.Tensor,
        lidar2img: torch.Tensor,
        img_hw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_min, y_min, z_min, x_max, y_max, z_max = self.pc_range
        reference_points_real = reference_points.clone()
        reference_points_real[..., 0:1] = reference_points_real[..., 0:1] * (x_max - x_min) + x_min
        reference_points_real[..., 1:2] = reference_points_real[..., 1:2] * (y_max - y_min) + y_min
        reference_points_real[..., 2:3] = reference_points_real[..., 2:3] * (z_max - z_min) + z_min

        homogeneous = torch.cat(
            (reference_points_real, torch.ones_like(reference_points_real[..., :1])),
            dim=-1,
        )  # [B, Q, D, 4]

        reference_points_cam = torch.einsum("bnij,bqdj->bnqdi", lidar2img, homogeneous)
        depth = reference_points_cam[..., 2:3]
        eps = 1e-5
        valid = depth > eps
        pixels = reference_points_cam[..., :2] / depth.clamp(min=eps)

        img_h = img_hw[..., 0].view(img_hw.shape[0], img_hw.shape[1], 1, 1, 1)
        img_w = img_hw[..., 1].view(img_hw.shape[0], img_hw.shape[1], 1, 1, 1)
        pixels[..., 0:1] = pixels[..., 0:1] / img_w
        pixels[..., 1:2] = pixels[..., 1:2] / img_h

        valid = (
            valid
            & (pixels[..., 0:1] > 0.0)
            & (pixels[..., 0:1] < 1.0)
            & (pixels[..., 1:2] > 0.0)
            & (pixels[..., 1:2] < 1.0)
        )
        pixels = torch.nan_to_num(pixels)
        valid = torch.nan_to_num(valid.to(dtype=torch.float32)).to(dtype=torch.bool)
        return pixels, valid.squeeze(-1)

    @staticmethod
    def _sample_single_level(
        level_features: torch.Tensor,
        reference_points_cam: torch.Tensor,
        visibility: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_cams, channels, _, _ = level_features.shape
        _, _, num_query, depth_bins, _ = reference_points_cam.shape

        features_flat = level_features.reshape(batch_size * num_cams, channels, level_features.shape[-2], level_features.shape[-1])
        grid = reference_points_cam.mul(2.0).sub(1.0).reshape(batch_size * num_cams, num_query, depth_bins, 2)

        sampled = F.grid_sample(
            features_flat,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )  # [B*N, C, Q, D]
        sampled = sampled.view(batch_size, num_cams, channels, num_query, depth_bins).permute(0, 1, 3, 4, 2)
        sampled = sampled * visibility.unsqueeze(-1).to(sampled.dtype)

        denom = visibility.sum(dim=1).clamp(min=1).unsqueeze(-1).to(sampled.dtype)
        return sampled.sum(dim=1) / denom  # [B, Q, D, C]

    def forward(
        self,
        camera_features: torch.Tensor | Sequence[torch.Tensor],
        volume_query: torch.Tensor,
        *,
        img_metas: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        levels = _as_camera_feature_levels(camera_features)
        if volume_query.dim() != 4:
            raise ValueError(
                f"volume_query must have shape [B, C, Hb, Wb], got {tuple(volume_query.shape)}."
            )
        batch_size, _, target_h, target_w = volume_query.shape

        first_level = levels[0]
        if first_level.dim() != 5:
            raise ValueError(
                "Each camera feature level must have shape [B, Ncam, C, H, W], "
                f"got {tuple(first_level.shape)}."
            )
        if first_level.shape[0] != batch_size:
            raise ValueError(
                f"camera_features batch size must match volume_query batch size={batch_size}, "
                f"got {first_level.shape[0]}."
            )
        num_cams = int(first_level.shape[1])
        if num_cams != self.num_cams:
            raise ValueError(
                f"Expected num_cams={self.num_cams} in camera_features, got {num_cams}."
            )

        for level_idx, level in enumerate(levels):
            if level.dim() != 5:
                raise ValueError(
                    f"camera_features[{level_idx}] must have shape [B, Ncam, C, H, W], "
                    f"got {tuple(level.shape)}."
                )
            if level.shape[0] != batch_size or level.shape[1] != num_cams:
                raise ValueError(
                    f"camera_features[{level_idx}] must match [B={batch_size}, Ncam={num_cams}], "
                    f"got {tuple(level.shape[:2])}."
                )

        lidar2img, img_hw = _validate_and_stack_metas(
            img_metas,
            batch_size=batch_size,
            num_cams=num_cams,
            device=volume_query.device,
            dtype=volume_query.dtype,
        )

        reference_points = self._build_reference_points(
            batch_size,
            height=target_h,
            width=target_w,
            depth=self.depth_bins,
            device=volume_query.device,
            dtype=volume_query.dtype,
        )
        reference_points_cam, visibility = self._point_sampling(reference_points, lidar2img, img_hw)

        fused_per_level: list[torch.Tensor] = []
        for level in levels:
            projected_level = self.level_projector(level)
            fused_per_level.append(
                self._sample_single_level(projected_level, reference_points_cam, visibility)
            )
        fused_volume_tokens = torch.stack(fused_per_level, dim=0).mean(dim=0)  # [B, Q, D, C]

        query_tokens = volume_query.flatten(2).transpose(1, 2)
        query_tokens = self.output_norm(self.query_proj(query_tokens))
        fused_volume_tokens = self.dropout(fused_volume_tokens + query_tokens.unsqueeze(2))

        volume_context = fused_volume_tokens.view(batch_size, target_h, target_w, self.depth_bins, self.embed_dims)
        volume_context = volume_context.permute(0, 4, 3, 1, 2).contiguous()  # [B, C, D, H, W]
        bev_context = volume_context.mean(dim=2)  # [B, C, H, W]

        camera_visibility = visibility.sum(dim=1).view(batch_size, target_h, target_w, self.depth_bins)
        camera_visibility = camera_visibility.permute(0, 3, 1, 2).contiguous()  # [B, D, H, W]

        return {
            "bev_context": bev_context,
            "volume_context": volume_context,
            "visibility": camera_visibility,
        }


class OccupancyHead(nn.Module):
    """Predict occupancy logits for each horizon and depth bin."""

    def __init__(self, embed_dims: int, occupancy_classes: int, depth_bins: int) -> None:
        super().__init__()
        self.embed_dims = int(embed_dims)
        self.occupancy_classes = int(occupancy_classes)
        self.depth_bins = int(depth_bins)
        self.refine = _conv_norm_act(embed_dims, embed_dims, stride=1)
        self.classifier = nn.Conv2d(
            embed_dims,
            self.occupancy_classes * self.depth_bins,
            kernel_size=1,
        )
        self.depth_context_proj = nn.Linear(self.embed_dims, self.occupancy_classes)

    def forward(
        self,
        horizon_features: torch.Tensor,
        *,
        depth_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, future_steps, channels, height, width = horizon_features.shape
        x = horizon_features.reshape(batch_size * future_steps, channels, height, width)
        x = self.refine(x)
        x = self.classifier(x)
        logits = x.view(
            batch_size,
            future_steps,
            self.occupancy_classes,
            self.depth_bins,
            height,
            width,
        )
        if depth_context is not None:
            expected_shape = (batch_size, future_steps, self.depth_bins, self.embed_dims)
            if depth_context.shape != expected_shape:
                raise ValueError(
                    f"depth_context must have shape {expected_shape}, got {tuple(depth_context.shape)}."
                )
            depth_bias = self.depth_context_proj(depth_context).permute(0, 1, 3, 2).unsqueeze(-1).unsqueeze(-1)
            logits = logits + depth_bias
        return logits


class TrajectoryHead(nn.Module):
    """Predict agent trajectories with integrated velocity consistency."""

    def __init__(self, embed_dims: int, dropout: float) -> None:
        super().__init__()
        self.agent_proj = nn.Linear(4, embed_dims)
        self.delta_mlp = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dims, 2),
        )

    def forward(
        self,
        agent_states: torch.Tensor,
        horizon_tokens: torch.Tensor,
        dt: float,
        *,
        time_deltas: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if agent_states.dim() != 3 or agent_states.shape[-1] != 4:
            raise ValueError(
                "Expected agent_states shape [B, Nagents, 4], "
                f"got {tuple(agent_states.shape)}."
            )
        batch_size, _, _ = agent_states.shape
        future_steps = horizon_tokens.shape[1]

        agent_embed = self.agent_proj(agent_states)  # [B, Nagents, C]
        fused = agent_embed[:, :, None, :] + horizon_tokens[:, None, :, :]
        delta_velocity = 0.1 * self.delta_mlp(fused)

        if time_deltas is None:
            step_deltas = agent_states.new_full((future_steps,), float(dt))
        else:
            step_deltas = time_deltas.to(device=agent_states.device, dtype=agent_states.dtype)
        if step_deltas.dim() == 1:
            if step_deltas.shape[0] != future_steps:
                raise ValueError(
                    f"time_deltas must have length={future_steps}, got {tuple(step_deltas.shape)}."
                )
            step_deltas = step_deltas.view(1, 1, future_steps, 1)
        elif step_deltas.dim() == 2:
            if step_deltas.shape != (batch_size, future_steps):
                raise ValueError(
                    f"time_deltas must have shape {(batch_size, future_steps)}, got {tuple(step_deltas.shape)}."
                )
            step_deltas = step_deltas.view(batch_size, 1, future_steps, 1)
        else:
            raise ValueError(f"time_deltas must be 1D or 2D, got {tuple(step_deltas.shape)}.")
        if torch.any(step_deltas <= 0):
            raise ValueError("time_deltas must be strictly positive.")

        base_velocity = agent_states[:, :, 2:4].unsqueeze(2)
        velocity = base_velocity + torch.cumsum(delta_velocity, dim=2)
        start_xy = agent_states[:, :, :2].unsqueeze(2)
        trajectory = start_xy + torch.cumsum(velocity * step_deltas, dim=2)
        return trajectory, velocity, delta_velocity


class SurroundOccPredictionLite(nn.Module):
    """Standalone SurroundOcc-style occupancy + trajectory predictor."""

    def __init__(self, cfg: SurroundOccPredictionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.spatial_encoder = SpatialEncoder(cfg.in_channels, cfg.embed_dims)
        self.temporal_encoder = TemporalContextEncoder(cfg.embed_dims, cfg.dropout)
        self.horizon_decoder = HorizonDecoder(cfg.embed_dims, cfg.future_steps, cfg.dropout)
        self.camera_view_fusion = CameraViewFusion(cfg)
        self.occupancy_head = OccupancyHead(
            cfg.embed_dims,
            cfg.occupancy_classes,
            cfg.depth_bins,
        )
        self.trajectory_head = TrajectoryHead(cfg.embed_dims, cfg.dropout)

    @staticmethod
    def _assert_finite_tensor(name: str, value: torch.Tensor) -> None:
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite values encountered in '{name}'.")

    def forward(
        self,
        history_bev: torch.Tensor,
        agent_states: torch.Tensor,
        *,
        camera_features: torch.Tensor | Sequence[torch.Tensor] | None = None,
        img_metas: list[dict[str, Any]] | None = None,
        history_time_indices: torch.Tensor | Sequence[float] | None = None,
        future_time_indices: torch.Tensor | Sequence[float] | None = None,
        decode: bool = False,
        occupancy_threshold: float = 0.5,
    ) -> dict[str, torch.Tensor] | dict[str, object]:
        if history_bev.dim() != 5:
            raise ValueError(
                "Expected history_bev shape [B, T, C, Hb, Wb], "
                f"got {tuple(history_bev.shape)}."
            )
        if agent_states.dim() != 3:
            raise ValueError(f"Expected agent_states shape [B, Nagents, 4], got {tuple(agent_states.shape)}.")
        if history_bev.shape[1] != self.cfg.history_steps:
            raise ValueError(
                f"Expected history_steps={self.cfg.history_steps}, got {history_bev.shape[1]}."
            )
        if history_bev.shape[2] != self.cfg.in_channels:
            raise ValueError(
                f"Expected in_channels={self.cfg.in_channels}, got {history_bev.shape[2]}."
            )
        if agent_states.shape[1] != self.cfg.num_agents:
            raise ValueError(
                f"Expected num_agents={self.cfg.num_agents}, got {agent_states.shape[1]}."
            )
        if agent_states.shape[2] != 4:
            raise ValueError(
                f"Expected agent_states last dimension = 4 for (x, y, vx, vy), got {agent_states.shape[2]}."
            )

        batch_size = history_bev.shape[0]
        history_times = _coerce_time_indices(
            history_time_indices,
            expected_steps=self.cfg.history_steps,
            device=history_bev.device,
            dtype=history_bev.dtype,
            name="history_time_indices",
        )
        default_future_indices: torch.Tensor | Sequence[float] | None = future_time_indices
        if default_future_indices is None:
            default_future_indices = (
                torch.arange(
                    1,
                    self.cfg.future_steps + 1,
                    device=history_bev.device,
                    dtype=history_bev.dtype,
                )
                * float(self.cfg.dt)
            )
        future_times = _coerce_time_indices(
            default_future_indices,
            expected_steps=self.cfg.future_steps,
            device=history_bev.device,
            dtype=history_bev.dtype,
            name="future_time_indices",
        )
        if history_times.dim() == 2 and history_times.shape[0] != batch_size:
            raise ValueError(
                "history_time_indices batch size must match history_bev batch size. "
                f"Got {history_times.shape[0]} vs {batch_size}."
            )
        if future_times.dim() == 2 and future_times.shape[0] != batch_size:
            raise ValueError(
                "future_time_indices batch size must match history_bev batch size. "
                f"Got {future_times.shape[0]} vs {batch_size}."
            )
        future_deltas = _time_deltas_from_indices(future_times)

        spatial_features = self.spatial_encoder(history_bev)
        temporal_sequence, temporal_context = self.temporal_encoder(
            spatial_features,
            history_time_indices=history_times,
        )
        horizon_features, horizon_tokens = self.horizon_decoder(
            temporal_context,
            spatial_features[:, -1],
            future_time_indices=future_times,
        )

        depth_context: torch.Tensor | None = None
        camera_visibility: torch.Tensor | None = None
        if camera_features is not None:
            if img_metas is None:
                raise ValueError(
                    "img_metas is required when camera_features are provided. "
                    "Required keys per sample: lidar2img, img_shape, pad_shape."
                )
            fusion_outputs = self.camera_view_fusion(
                camera_features,
                spatial_features[:, -1],
                img_metas=img_metas,
            )
            horizon_features = horizon_features + fusion_outputs["bev_context"].unsqueeze(1)
            depth_tokens = fusion_outputs["volume_context"].mean(dim=(-1, -2)).permute(0, 2, 1)
            depth_context = horizon_tokens[:, :, None, :] + depth_tokens[:, None, :, :]
            camera_visibility = fusion_outputs["visibility"]

        occupancy_logits = self.occupancy_head(horizon_features, depth_context=depth_context)
        trajectory, velocity, delta_velocity = self.trajectory_head(
            agent_states,
            horizon_tokens,
            dt=self.cfg.dt,
            time_deltas=future_deltas,
        )

        resolved_future_times = (
            future_times if future_times.dim() == 2 else future_times.unsqueeze(0).expand(batch_size, -1)
        )
        outputs: dict[str, torch.Tensor] = {
            "occupancy_logits": occupancy_logits,
            "trajectory": trajectory,
            "velocity": velocity,
            "delta_velocity": delta_velocity,
            "horizon_tokens": horizon_tokens,
            "temporal_sequence": temporal_sequence,
            "future_time_indices": resolved_future_times,
        }
        if camera_visibility is not None:
            outputs["camera_visibility"] = camera_visibility

        for name in (
            "occupancy_logits",
            "trajectory",
            "velocity",
            "delta_velocity",
            "horizon_tokens",
            "temporal_sequence",
            "future_time_indices",
        ):
            self._assert_finite_tensor(name, outputs[name])

        if decode:
            decoded = decode_predictions(outputs, occupancy_threshold=occupancy_threshold)
            return {"preds": outputs, "decoded": decoded}
        return outputs
