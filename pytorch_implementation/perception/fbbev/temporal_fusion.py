"""Temporal BEV fusion for FB-BEV."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import FBBEVForwardConfig


class TemporalFusionLite(nn.Module):
    """Fuse current BEV with history using upstream-style temporal updates."""

    def __init__(self, cfg: FBBEVForwardConfig) -> None:
        super().__init__()
        self.history_cat_num = int(cfg.history_cat_num)
        self.history_cam_sweep_freq = float(cfg.history_cam_sweep_freq)
        self.interpolation_mode = str(cfg.history_interpolation_mode)
        self.pc_range = tuple(float(v) for v in cfg.pc_range)
        channels = int(cfg.embed_dims)
        self.time_conv = nn.Sequential(
            nn.Conv2d(channels + 1, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.cat_conv = nn.Sequential(
            nn.Conv2d(channels * (self.history_cat_num + 1), channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.history_bev: torch.Tensor | None = None  # [B, T, C, H, W]
        self.history_sweep_time: torch.Tensor | None = None  # [B, T]
        self.history_seq_ids: torch.Tensor | None = None  # [B]
        self.history_forward_augs: torch.Tensor | None = None  # [B, 4, 4]

    def _extract_sequence_ids(self, img_metas: list[dict], device: torch.device) -> torch.Tensor:
        seq_ids = []
        for idx, meta in enumerate(img_metas):
            if "sequence_group_idx" not in meta:
                raise KeyError(f"img_metas[{idx}] is missing required key 'sequence_group_idx'.")
            seq_ids.append(int(meta["sequence_group_idx"]))
        return torch.as_tensor(seq_ids, dtype=torch.long, device=device)

    def _extract_start_flags(self, img_metas: list[dict], device: torch.device) -> torch.Tensor:
        flags = []
        for idx, meta in enumerate(img_metas):
            if "start_of_sequence" not in meta:
                raise KeyError(f"img_metas[{idx}] is missing required key 'start_of_sequence'.")
            flags.append(bool(meta["start_of_sequence"]))
        return torch.as_tensor(flags, dtype=torch.bool, device=device)

    def _extract_curr_to_prev(self, img_metas: list[dict], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mats = []
        for idx, meta in enumerate(img_metas):
            if "curr_to_prev_ego_rt" not in meta:
                raise KeyError(f"img_metas[{idx}] is missing required key 'curr_to_prev_ego_rt'.")
            mat = torch.as_tensor(meta["curr_to_prev_ego_rt"], dtype=dtype, device=device)
            if mat.shape != (4, 4):
                raise ValueError(f"img_metas[{idx}]['curr_to_prev_ego_rt'] must be 4x4, got {tuple(mat.shape)}")
            if not torch.isfinite(mat).all():
                raise ValueError(f"img_metas[{idx}]['curr_to_prev_ego_rt'] must be finite.")
            mats.append(mat)
        return torch.stack(mats, dim=0)

    def _extract_forward_augs(self, img_metas: list[dict], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mats = []
        identity = torch.eye(4, dtype=dtype, device=device)
        for meta in img_metas:
            bda = meta.get("bda")
            if bda is None:
                mats.append(identity.clone())
                continue
            bda_tensor = torch.as_tensor(bda, dtype=dtype, device=device)
            if bda_tensor.shape == (3, 3):
                mat = identity.clone()
                mat[:3, :3] = bda_tensor
                mats.append(mat)
            elif bda_tensor.shape == (4, 4):
                mats.append(bda_tensor)
            else:
                raise ValueError(f"bda must be 3x3 or 4x4, got {tuple(bda_tensor.shape)}")
        return torch.stack(mats, dim=0)

    def _to_hom2d(self, mat: torch.Tensor) -> torch.Tensor:
        if mat.shape[-2:] == (4, 4):
            out = torch.eye(3, dtype=mat.dtype, device=mat.device).view(1, 3, 3).repeat(mat.shape[0], 1, 1)
            out[:, :2, :2] = mat[:, :2, :2]
            out[:, :2, 2] = mat[:, :2, 3]
            return out
        if mat.shape[-2:] == (3, 3):
            return mat
        raise ValueError(f"Expected batched 4x4 or 3x3 matrices, got {tuple(mat.shape)}")

    def _build_feat2bev(self, *, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        dx = (self.pc_range[3] - self.pc_range[0]) / max(w, 1)
        dy = (self.pc_range[4] - self.pc_range[1]) / max(h, 1)
        feat2bev = torch.eye(3, dtype=dtype, device=device)
        feat2bev[0, 0] = dx
        feat2bev[1, 1] = dy
        feat2bev[0, 2] = self.pc_range[0]
        feat2bev[1, 2] = self.pc_range[1]
        return feat2bev

    def _generate_grid(
        self,
        history_forward_augs: torch.Tensor,
        forward_augs: torch.Tensor,
        curr_to_prev_ego_rt: torch.Tensor,
        *,
        h: int,
        w: int,
    ) -> torch.Tensor:
        batch_size = curr_to_prev_ego_rt.shape[0]
        feat2bev = self._build_feat2bev(h=h, w=w, device=curr_to_prev_ego_rt.device, dtype=curr_to_prev_ego_rt.dtype)
        feat2bev = feat2bev.view(1, 3, 3).expand(batch_size, -1, -1)
        history_aug_2d = self._to_hom2d(history_forward_augs)
        forward_aug_2d = self._to_hom2d(forward_augs)
        curr_to_prev_2d = self._to_hom2d(curr_to_prev_ego_rt)
        rt_flow = (
            torch.linalg.inv(feat2bev)
            .matmul(history_aug_2d)
            .matmul(curr_to_prev_2d)
            .matmul(torch.linalg.inv(forward_aug_2d))
            .matmul(feat2bev)
        )

        xs = torch.linspace(0.0, max(float(w - 1), 0.0), w, dtype=curr_to_prev_ego_rt.dtype, device=curr_to_prev_ego_rt.device)
        ys = torch.linspace(0.0, max(float(h - 1), 0.0), h, dtype=curr_to_prev_ego_rt.dtype, device=curr_to_prev_ego_rt.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        base = torch.stack((grid_x, grid_y, torch.ones_like(grid_x)), dim=-1)  # [H, W, 3]
        transformed = torch.einsum("bij,hwj->bhwi", rt_flow, base)
        x = transformed[..., 0]
        y = transformed[..., 1]
        x_norm = (x / max(float(w - 1), 1.0)) * 2.0 - 1.0
        y_norm = (y / max(float(h - 1), 1.0)) * 2.0 - 1.0
        return torch.stack((x_norm, y_norm), dim=-1)

    def _warp_history(self, history: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels, h, w = history.shape
        history_flat = history.reshape(batch_size * time_steps, channels, h, w)
        grid_repeat = grid[:, None].repeat(1, time_steps, 1, 1, 1).reshape(batch_size * time_steps, h, w, 2)
        warped = F.grid_sample(
            history_flat,
            grid_repeat,
            mode=self.interpolation_mode,
            padding_mode="zeros",
            align_corners=True,
        )
        return warped.view(batch_size, time_steps, channels, h, w)

    def _reset_history(self) -> None:
        self.history_bev = None
        self.history_sweep_time = None
        self.history_seq_ids = None
        self.history_forward_augs = None

    def _ensure_history(self, curr_bev: torch.Tensor, seq_ids: torch.Tensor, forward_augs: torch.Tensor) -> None:
        batch_size, channels, height, width = curr_bev.shape
        if self.history_bev is None:
            self.history_bev = curr_bev.unsqueeze(1).repeat(1, self.history_cat_num, 1, 1, 1).detach()
            self.history_sweep_time = curr_bev.new_zeros(batch_size, self.history_cat_num)
            self.history_seq_ids = seq_ids.clone()
            self.history_forward_augs = forward_augs.clone()
            return
        if self.history_bev.shape[:1] != (batch_size,) or self.history_bev.shape[-2:] != (height, width):
            self._reset_history()
            self._ensure_history(curr_bev, seq_ids, forward_augs)

    def forward(self, curr_bev: torch.Tensor, img_metas: list[dict]) -> torch.Tensor:
        """Args:
        curr_bev: [B, C, H, W]
        """

        if curr_bev.dim() != 4:
            raise ValueError(f"Expected curr_bev [B, C, H, W], got {tuple(curr_bev.shape)}")
        if not isinstance(img_metas, list):
            raise TypeError(f"img_metas must be list[dict], got {type(img_metas)}")
        if len(img_metas) != curr_bev.shape[0]:
            raise ValueError(f"img_metas length {len(img_metas)} must equal batch size {curr_bev.shape[0]}.")

        device = curr_bev.device
        dtype = curr_bev.dtype
        seq_ids = self._extract_sequence_ids(img_metas, device)
        start_flags = self._extract_start_flags(img_metas, device)
        curr_to_prev = self._extract_curr_to_prev(img_metas, device, dtype)
        forward_augs = self._extract_forward_augs(img_metas, device, dtype)

        self._ensure_history(curr_bev, seq_ids, forward_augs)
        assert self.history_bev is not None
        assert self.history_sweep_time is not None
        assert self.history_seq_ids is not None
        assert self.history_forward_augs is not None

        mismatch = self.history_seq_ids != seq_ids
        if bool((mismatch & (~start_flags)).any()):
            raise AssertionError(
                "History sequence ids must match current sequence ids for non-start frames."
            )

        self.history_sweep_time = self.history_sweep_time + 1.0
        reset_mask = start_flags
        if reset_mask.any():
            self.history_bev[reset_mask] = curr_bev[reset_mask].unsqueeze(1).repeat(1, self.history_cat_num, 1, 1, 1)
            self.history_sweep_time[reset_mask] = 0.0
            self.history_seq_ids[reset_mask] = seq_ids[reset_mask]
            self.history_forward_augs[reset_mask] = forward_augs[reset_mask]

        _, _, h, w = curr_bev.shape
        grid = self._generate_grid(
            self.history_forward_augs,
            forward_augs,
            curr_to_prev,
            h=h,
            w=w,
        )
        warped_history = self._warp_history(self.history_bev, grid)
        sweep_time = self.history_sweep_time
        feats = torch.cat([curr_bev.unsqueeze(1), warped_history], dim=1)  # [B, T+1, C, H, W]

        time_embed = torch.cat([sweep_time.new_zeros(sweep_time.shape[0], 1), sweep_time], dim=1)
        time_embed = time_embed * self.history_cam_sweep_freq
        time_map = time_embed[:, :, None, None, None].expand(-1, -1, 1, curr_bev.shape[-2], curr_bev.shape[-1])
        feats_with_time = torch.cat([feats, time_map], dim=2)

        fused_steps = self.time_conv(feats_with_time.reshape(-1, feats_with_time.shape[2], *feats_with_time.shape[3:]))
        fused_steps = fused_steps.reshape(curr_bev.shape[0], self.history_cat_num + 1, curr_bev.shape[1], *curr_bev.shape[-2:])
        fused = self.cat_conv(fused_steps.reshape(curr_bev.shape[0], -1, *curr_bev.shape[-2:]))

        new_history = torch.cat([curr_bev.unsqueeze(1), warped_history], dim=1)[:, : self.history_cat_num]
        new_time = torch.cat([sweep_time.new_zeros(sweep_time.shape[0], 1), sweep_time], dim=1)[:, : self.history_cat_num]
        self.history_bev = new_history.detach()
        self.history_sweep_time = new_time.detach()
        self.history_seq_ids = seq_ids.detach()
        self.history_forward_augs = forward_augs.detach()
        return fused

    def clear(self) -> None:
        self._reset_history()
