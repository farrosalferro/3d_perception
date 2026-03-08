"""Temporal BEV fusion for FB-BEV."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from .config import FBBEVForwardConfig


class TemporalFusionLite(nn.Module):
    """Fuse current BEV with a short history queue."""

    def __init__(self, cfg: FBBEVForwardConfig) -> None:
        super().__init__()
        self.history_cat_num = cfg.history_cat_num
        self.history_cam_sweep_freq = float(cfg.history_cam_sweep_freq)
        self.voxel_size = cfg.voxel_size
        channels = cfg.embed_dims
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

    def _extract_sequence_ids(self, img_metas: list[dict], device: torch.device) -> torch.Tensor:
        seq_ids = [int(meta.get("sequence_group_idx", idx)) for idx, meta in enumerate(img_metas)]
        return torch.as_tensor(seq_ids, dtype=torch.long, device=device)

    def _extract_start_flags(self, img_metas: list[dict], device: torch.device) -> torch.Tensor:
        flags = [bool(meta.get("start_of_sequence", True)) for meta in img_metas]
        return torch.as_tensor(flags, dtype=torch.bool, device=device)

    def _extract_curr_to_prev(self, img_metas: list[dict], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mats = []
        for meta in img_metas:
            mat = meta.get("curr_to_prev_ego_rt")
            if mat is None:
                mats.append(torch.eye(4, dtype=dtype, device=device))
            else:
                mats.append(torch.as_tensor(mat, dtype=dtype, device=device))
        return torch.stack(mats, dim=0)

    def _warp_single(self, feat: torch.Tensor, curr_to_prev: torch.Tensor) -> torch.Tensor:
        """Approximate ego-motion alignment with integer grid shifts."""

        shift_x = int(torch.round(curr_to_prev[0, 3] / self.voxel_size[0]).item())
        shift_y = int(torch.round(curr_to_prev[1, 3] / self.voxel_size[1]).item())
        return torch.roll(feat, shifts=(shift_y, shift_x), dims=(-2, -1))

    def _warp_history(self, history: torch.Tensor, curr_to_prev: torch.Tensor) -> torch.Tensor:
        warped = []
        for batch_idx in range(history.shape[0]):
            cur_mat = curr_to_prev[batch_idx]
            warped_steps = [self._warp_single(history[batch_idx, t], cur_mat) for t in range(history.shape[1])]
            warped.append(torch.stack(warped_steps, dim=0))
        return torch.stack(warped, dim=0)

    def _reset_history(self) -> None:
        self.history_bev = None
        self.history_sweep_time = None
        self.history_seq_ids = None

    def _ensure_history(self, curr_bev: torch.Tensor, seq_ids: torch.Tensor) -> None:
        batch_size, channels, height, width = curr_bev.shape
        if self.history_bev is None:
            self.history_bev = curr_bev.unsqueeze(1).repeat(1, self.history_cat_num, 1, 1, 1).detach()
            self.history_sweep_time = curr_bev.new_zeros(batch_size, self.history_cat_num)
            self.history_seq_ids = seq_ids.clone()
            return
        if self.history_bev.shape[:1] != (batch_size,) or self.history_bev.shape[-2:] != (height, width):
            self._reset_history()
            self._ensure_history(curr_bev, seq_ids)

    def forward(self, curr_bev: torch.Tensor, img_metas: list[dict]) -> torch.Tensor:
        """Args:
        curr_bev: [B, C, H, W]
        """

        device = curr_bev.device
        dtype = curr_bev.dtype
        seq_ids = self._extract_sequence_ids(img_metas, device)
        start_flags = self._extract_start_flags(img_metas, device)
        curr_to_prev = self._extract_curr_to_prev(img_metas, device, dtype)

        self._ensure_history(curr_bev, seq_ids)
        assert self.history_bev is not None
        assert self.history_sweep_time is not None
        assert self.history_seq_ids is not None

        mismatch = self.history_seq_ids != seq_ids
        reset_mask = start_flags | mismatch
        if reset_mask.any():
            self.history_bev[reset_mask] = curr_bev[reset_mask].unsqueeze(1).repeat(1, self.history_cat_num, 1, 1, 1)
            self.history_sweep_time[reset_mask] = 0.0
            self.history_seq_ids[reset_mask] = seq_ids[reset_mask]

        warped_history = self._warp_history(self.history_bev, curr_to_prev)
        sweep_time = self.history_sweep_time + 1.0
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
        return fused

    def clear(self) -> None:
        self._reset_history()
