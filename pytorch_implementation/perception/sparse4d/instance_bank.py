"""Sparse4D-style instance bank with temporal cache/update semantics."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ = list(range(11))


def topk(
    confidence: torch.Tensor,
    k: int,
    *inputs: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Gather top-k entries on dimension-1 for batched tensors."""

    batch_size, num_items = confidence.shape[:2]
    k = max(0, min(int(k), int(num_items)))
    if k == 0:
        return confidence[:, :0], [x[:, :0] for x in inputs]

    top_confidence, indices = torch.topk(confidence, k, dim=1)
    flat_indices = (
        indices + torch.arange(batch_size, device=indices.device)[:, None] * num_items
    ).reshape(-1)

    outputs: list[torch.Tensor] = []
    for value in inputs:
        flat_value = value.flatten(end_dim=1)
        gathered = flat_value.index_select(0, flat_indices)
        target_shape = (batch_size, k) + value.shape[2:]
        outputs.append(gathered.reshape(target_shape))
    return top_confidence, outputs


class InstanceBankLite(nn.Module):
    """Learnable anchors/features plus temporal cached instance management."""

    def __init__(
        self,
        num_queries: int,
        embed_dims: int,
        box_code_size: int,
        *,
        num_temp_instances: int = 0,
        default_time_interval: float = 0.5,
        confidence_decay: float = 0.6,
        max_time_interval: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_queries = int(num_queries)
        self.num_anchor = int(num_queries)
        self.embed_dims = int(embed_dims)
        self.box_code_size = int(box_code_size)
        self.num_temp_instances = max(0, min(int(num_temp_instances), self.num_anchor))
        self.default_time_interval = float(default_time_interval)
        self.confidence_decay = float(confidence_decay)
        self.max_time_interval = float(max_time_interval)

        self.instance_feature = nn.Parameter(torch.zeros(self.num_anchor, self.embed_dims))
        self.anchor = nn.Parameter(torch.zeros(self.num_anchor, self.box_code_size))
        self.anchor_init = self.anchor.detach().clone()
        self._init_parameters()
        self.reset()

    def _init_parameters(self) -> None:
        nn.init.xavier_uniform_(self.instance_feature)
        with torch.no_grad():
            self.anchor.zero_()
            if self.box_code_size >= 8:
                self.anchor[:, COS_YAW] = 1.0
            elif self.box_code_size >= 6:
                self.anchor[:, 3:6] = 1.0
        self.anchor_init = self.anchor.detach().clone()

    def init_weight(self) -> None:
        with torch.no_grad():
            self.anchor.copy_(self.anchor_init.to(device=self.anchor.device, dtype=self.anchor.dtype))
            nn.init.xavier_uniform_(self.instance_feature)

    def reset(self) -> None:
        self.cached_feature: torch.Tensor | None = None
        self.cached_anchor: torch.Tensor | None = None
        self.metas: Any = None
        self.mask: torch.Tensor | None = None
        self.confidence: torch.Tensor | None = None
        self.temp_confidence: torch.Tensor | None = None
        self.instance_id: torch.Tensor | None = None
        self.prev_id = 0

    @staticmethod
    def _extract_timestamp(
        metas: Any,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if metas is None:
            return None
        timestamp = None
        if isinstance(metas, dict):
            timestamp = metas.get("timestamp")
        elif isinstance(metas, list):
            values = []
            for sample_meta in metas:
                if not isinstance(sample_meta, dict) or "timestamp" not in sample_meta:
                    return None
                values.append(sample_meta["timestamp"])
            timestamp = values
        if timestamp is None:
            return None
        timestamp_tensor = torch.as_tensor(timestamp, device=device, dtype=dtype).reshape(-1)
        if timestamp_tensor.numel() == 1:
            timestamp_tensor = timestamp_tensor.expand(batch_size)
        if timestamp_tensor.numel() != batch_size:
            return None
        return timestamp_tensor

    @staticmethod
    def _extract_img_metas(metas: Any) -> list[dict[str, Any]] | None:
        if isinstance(metas, dict):
            img_metas = metas.get("img_metas")
            if isinstance(img_metas, list):
                return img_metas
            return None
        if isinstance(metas, list):
            return metas if all(isinstance(x, dict) for x in metas) else None
        return None

    @staticmethod
    def _build_temp2cur_transform(
        current_metas: Any,
        history_metas: Any,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        cur_img_metas = InstanceBankLite._extract_img_metas(current_metas)
        hist_img_metas = InstanceBankLite._extract_img_metas(history_metas)
        if cur_img_metas is None or hist_img_metas is None:
            return None
        if len(cur_img_metas) < batch_size or len(hist_img_metas) < batch_size:
            return None

        transforms = []
        for idx in range(batch_size):
            cur_meta = cur_img_metas[idx]
            hist_meta = hist_img_metas[idx]
            cur_global_inv = cur_meta.get("T_global_inv")
            hist_global = hist_meta.get("T_global")
            if cur_global_inv is None or hist_global is None:
                return None
            cur_global_inv = torch.as_tensor(cur_global_inv, device=device, dtype=dtype)
            hist_global = torch.as_tensor(hist_global, device=device, dtype=dtype)
            transforms.append(cur_global_inv @ hist_global)
        return torch.stack(transforms, dim=0)

    @staticmethod
    def _ensure_homogeneous(transform: torch.Tensor) -> torch.Tensor:
        if transform.shape[-2:] == (4, 4):
            return transform
        if transform.shape[-2:] == (3, 4):
            pad = torch.zeros(
                transform.shape[:-2] + (1, 4),
                device=transform.device,
                dtype=transform.dtype,
            )
            pad[..., 0, 3] = 1.0
            return torch.cat([transform, pad], dim=-2)
        raise ValueError("Expected transform shape [..., 4, 4] or [..., 3, 4].")

    @staticmethod
    def _project_anchors(
        anchor: torch.Tensor,
        transform: torch.Tensor,
        *,
        time_interval: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project anchor states to a new frame, mirroring upstream contracts."""

        batch_size, _, box_code_size = anchor.shape
        transform = InstanceBankLite._ensure_homogeneous(transform).to(
            device=anchor.device, dtype=anchor.dtype
        )

        center = anchor[..., [X, Y, Z]]
        vel_dim = max(0, box_code_size - VX)
        if time_interval is not None and vel_dim > 0:
            interval = time_interval.to(device=anchor.device, dtype=anchor.dtype).reshape(batch_size, 1, 1)
            translation = center.new_zeros(center.shape)
            translation[..., :vel_dim] = anchor[..., VX : VX + vel_dim] * interval
            center = center - translation

        rot = transform[:, None, :3, :3]
        trans = transform[:, None, :3, 3]
        center = torch.matmul(rot, center.unsqueeze(-1)).squeeze(-1) + trans

        pieces = [center]
        if box_code_size > 3:
            size_end = min(box_code_size, H + 1)
            if size_end > 3:
                pieces.append(anchor[..., 3:size_end])
        if box_code_size > SIN_YAW:
            if box_code_size > COS_YAW:
                yaw = torch.matmul(
                    transform[:, None, :2, :2],
                    anchor[..., [COS_YAW, SIN_YAW]].unsqueeze(-1),
                ).squeeze(-1)
                # Keep upstream order ([cos, sin]) for behavior parity.
                pieces.append(yaw)
            else:
                pieces.append(anchor[..., SIN_YAW : SIN_YAW + 1])
        if vel_dim > 0:
            vel = torch.matmul(
                transform[:, None, :vel_dim, :vel_dim],
                anchor[..., VX : VX + vel_dim].unsqueeze(-1),
            ).squeeze(-1)
            pieces.append(vel)

        projected = torch.cat(pieces, dim=-1)
        if projected.shape[-1] < box_code_size:
            projected = F.pad(projected, (0, box_code_size - projected.shape[-1]))
        elif projected.shape[-1] > box_code_size:
            projected = projected[..., :box_code_size]
        return projected

    def get(
        self,
        batch_size: int,
        metas: Any = None,
        dn_metas: dict[str, torch.Tensor] | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
    ]:
        if device is None:
            device = self.instance_feature.device
        if dtype is None:
            dtype = self.instance_feature.dtype

        instance_feature = self.instance_feature[None].expand(batch_size, -1, -1).to(
            device=device, dtype=dtype
        )
        anchor = self.anchor[None].expand(batch_size, -1, -1).to(device=device, dtype=dtype)

        has_cache = (
            self.cached_anchor is not None
            and self.cached_feature is not None
            and batch_size == self.cached_anchor.shape[0]
        )
        if has_cache:
            current_timestamp = self._extract_timestamp(
                metas,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            history_timestamp = self._extract_timestamp(
                self.metas,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            if current_timestamp is not None and history_timestamp is not None:
                time_interval = current_timestamp - history_timestamp
                self.mask = torch.abs(time_interval) <= self.max_time_interval
                valid_interval = torch.logical_and(time_interval != 0, self.mask)
                default_interval = time_interval.new_full((), self.default_time_interval)
                time_interval = torch.where(valid_interval, time_interval, default_interval)
            else:
                time_interval = instance_feature.new_full((batch_size,), self.default_time_interval)
                self.mask = torch.ones(batch_size, device=device, dtype=torch.bool)

            temp2cur = self._build_temp2cur_transform(
                metas,
                self.metas,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            if temp2cur is not None:
                self.cached_anchor = self._project_anchors(
                    self.cached_anchor.to(device=device, dtype=dtype),
                    temp2cur,
                    time_interval=-time_interval,
                )
                if (
                    dn_metas is not None
                    and isinstance(dn_metas, dict)
                    and "dn_anchor" in dn_metas
                    and torch.is_tensor(dn_metas["dn_anchor"])
                    and dn_metas["dn_anchor"].shape[0] == batch_size
                ):
                    dn_anchor = dn_metas["dn_anchor"].to(device=device, dtype=dtype)
                    num_dn_group, num_dn = dn_anchor.shape[1:3]
                    dn_flat = dn_anchor.flatten(1, 2)
                    dn_projected = self._project_anchors(
                        dn_flat,
                        temp2cur,
                        time_interval=-time_interval,
                    )
                    dn_metas["dn_anchor"] = dn_projected.reshape(
                        batch_size, num_dn_group, num_dn, -1
                    )
        else:
            self.reset()
            time_interval = instance_feature.new_full((batch_size,), self.default_time_interval)
            self.mask = torch.zeros(batch_size, device=device, dtype=torch.bool)

        return (
            instance_feature,
            anchor,
            self.cached_feature,
            self.cached_anchor,
            time_interval,
        )

    def forward(
        self,
        batch_size: int,
        *,
        metas: Any = None,
        dn_metas: dict[str, torch.Tensor] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
    ]:
        return self.get(
            batch_size=batch_size,
            metas=metas,
            dn_metas=dn_metas,
            device=device,
            dtype=dtype,
        )

    def update(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cached_feature is None or self.cached_anchor is None:
            return instance_feature, anchor

        num_dn = 0
        if instance_feature.shape[1] > self.num_anchor:
            num_dn = instance_feature.shape[1] - self.num_anchor
            dn_instance_feature = instance_feature[:, -num_dn:]
            dn_anchor = anchor[:, -num_dn:]
            instance_feature = instance_feature[:, : self.num_anchor]
            anchor = anchor[:, : self.num_anchor]
            confidence = confidence[:, : self.num_anchor]

        if confidence.dim() == 3:
            confidence = confidence.max(dim=-1).values

        num_dynamic = max(self.num_anchor - self.num_temp_instances, 0)
        _, (selected_feature, selected_anchor) = topk(
            confidence,
            num_dynamic,
            instance_feature,
            anchor,
        )

        selected_feature = torch.cat([self.cached_feature, selected_feature], dim=1)
        selected_anchor = torch.cat([self.cached_anchor, selected_anchor], dim=1)
        if selected_feature.shape[1] < instance_feature.shape[1]:
            selected_feature = torch.cat(
                [selected_feature, instance_feature[:, selected_feature.shape[1] :]],
                dim=1,
            )
            selected_anchor = torch.cat(
                [selected_anchor, anchor[:, selected_anchor.shape[1] :]],
                dim=1,
            )
        selected_feature = selected_feature[:, : instance_feature.shape[1]]
        selected_anchor = selected_anchor[:, : anchor.shape[1]]

        mask = self.mask
        if mask is None:
            mask = torch.ones(
                instance_feature.shape[0],
                device=instance_feature.device,
                dtype=torch.bool,
            )
        instance_feature = torch.where(mask[:, None, None], selected_feature, instance_feature)
        anchor = torch.where(mask[:, None, None], selected_anchor, anchor)

        if self.instance_id is not None:
            invalid = self.instance_id.new_full(self.instance_id.shape, -1)
            self.instance_id = torch.where(mask[:, None], self.instance_id, invalid)

        if num_dn > 0:
            instance_feature = torch.cat([instance_feature, dn_instance_feature], dim=1)
            anchor = torch.cat([anchor, dn_anchor], dim=1)
        return instance_feature, anchor

    def cache(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        confidence: torch.Tensor,
        metas: Any = None,
        feature_maps: list[torch.Tensor] | None = None,
    ) -> None:
        del feature_maps  # Kept for interface compatibility.
        if self.num_temp_instances <= 0:
            return

        instance_feature = instance_feature.detach()
        anchor = anchor.detach()
        confidence = confidence.detach()

        self.metas = metas
        if confidence.dim() == 3:
            confidence = confidence.max(dim=-1).values.sigmoid()
        else:
            confidence = confidence.sigmoid()
        if self.confidence is not None and self.confidence.shape[1] > 0:
            keep = min(self.num_temp_instances, self.confidence.shape[1], confidence.shape[1])
            confidence[:, :keep] = torch.maximum(
                self.confidence[:, :keep] * self.confidence_decay,
                confidence[:, :keep],
            )
        self.temp_confidence = confidence
        self.confidence, (self.cached_feature, self.cached_anchor) = topk(
            confidence,
            self.num_temp_instances,
            instance_feature,
            anchor,
        )

    def get_instance_id(
        self,
        confidence: torch.Tensor,
        anchor: torch.Tensor | None = None,
        threshold: float | None = None,
    ) -> torch.Tensor:
        del anchor  # The upstream interface accepts anchor, but id assignment uses confidence.
        if confidence.dim() == 3:
            confidence = confidence.max(dim=-1).values.sigmoid()
        else:
            confidence = confidence.sigmoid()
        instance_id = confidence.new_full(confidence.shape, -1, dtype=torch.long)

        if self.instance_id is not None and self.instance_id.shape[0] == instance_id.shape[0]:
            keep = min(self.instance_id.shape[1], instance_id.shape[1])
            instance_id[:, :keep] = self.instance_id[:, :keep]

        mask = instance_id < 0
        if threshold is not None:
            mask = torch.logical_and(mask, confidence >= threshold)
        num_new_instance = int(mask.sum().item())
        if num_new_instance > 0:
            new_ids = torch.arange(
                num_new_instance,
                device=instance_id.device,
                dtype=instance_id.dtype,
            ) + int(self.prev_id)
            instance_id[mask] = new_ids
            self.prev_id += num_new_instance
        if self.num_temp_instances > 0:
            self.update_instance_id(instance_id=instance_id, confidence=confidence)
        return instance_id

    def update_instance_id(
        self,
        instance_id: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
    ) -> None:
        if self.num_temp_instances <= 0:
            return
        if instance_id is None:
            if self.instance_id is None:
                return
            instance_id = self.instance_id
        if self.temp_confidence is None:
            if confidence is None:
                return
            if confidence.dim() == 3:
                temp_conf = confidence.max(dim=-1).values
            else:
                temp_conf = confidence
        else:
            temp_conf = self.temp_confidence
        if temp_conf.shape[:2] != instance_id.shape[:2]:
            keep = min(temp_conf.shape[1], instance_id.shape[1])
            temp_conf = temp_conf[:, :keep]
            instance_id = instance_id[:, :keep]

        k = min(self.num_temp_instances, instance_id.shape[1])
        if k <= 0:
            self.instance_id = instance_id.new_full(
                (instance_id.shape[0], self.num_anchor),
                -1,
            )
            return
        selected_instance = topk(temp_conf, k, instance_id)[1][0].squeeze(-1)
        if selected_instance.dim() == 1:
            selected_instance = selected_instance.unsqueeze(0)
        self.instance_id = F.pad(
            selected_instance,
            (0, max(0, self.num_anchor - k)),
            value=-1,
        )
        self.instance_id = self.instance_id[:, : self.num_anchor]
