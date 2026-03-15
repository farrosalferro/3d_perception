"""Temporal neck semantics for pure-PyTorch BEVerse-lite."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .config import BEVerseForwardConfig


class TemporalPredictorLite(nn.Module):
    """Decode a fixed horizon of temporal tokens from BEV context."""

    def __init__(self, cfg: BEVerseForwardConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.receptive_field = max(1, int(cfg.temporal_receptive_field))
        self.init_proj = nn.Linear(cfg.embed_dims, cfg.embed_dims)
        self.history_proj = nn.Linear(cfg.embed_dims, cfg.embed_dims)
        self.time_embedding = nn.Embedding(cfg.pred_horizon, cfg.embed_dims)
        self.gru = nn.GRU(input_size=cfg.embed_dims, hidden_size=cfg.embed_dims, batch_first=True)

    def _as_sequence(self, bev_embed: torch.Tensor) -> torch.Tensor:
        if bev_embed.dim() == 4:
            bev_embed = bev_embed.unsqueeze(1)
        if bev_embed.dim() != 5:
            raise ValueError(
                f"Expected BEV features [B, C, H, W] or [B, S, C, H, W], got {tuple(bev_embed.shape)}"
            )
        if bev_embed.shape[2] != self.cfg.embed_dims:
            raise ValueError(
                f"Expected BEV channel dim {self.cfg.embed_dims}, got {bev_embed.shape[2]}"
            )
        return bev_embed

    def _validate_future_egomotion(
        self,
        future_egomotion: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if future_egomotion is None:
            return None
        if future_egomotion.dim() != 3:
            raise ValueError(
                f"future_egomotion must be [B, S, D], got {tuple(future_egomotion.shape)}"
            )
        if future_egomotion.shape[0] != batch_size:
            raise ValueError(
                f"future_egomotion batch mismatch: expected {batch_size}, got {future_egomotion.shape[0]}"
            )
        if future_egomotion.shape[1] < seq_len:
            raise ValueError(
                "future_egomotion must provide at least one motion vector per history step."
            )
        if future_egomotion.shape[2] < 2:
            raise ValueError("future_egomotion must provide at least XY translation components.")
        return future_egomotion[:, :seq_len].to(device=device)

    def _validate_img_is_valid(
        self,
        img_is_valid: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if img_is_valid is None:
            return None
        if img_is_valid.dim() != 2:
            raise ValueError(f"img_is_valid must be [B, S], got {tuple(img_is_valid.shape)}")
        if img_is_valid.shape[0] != batch_size:
            raise ValueError(
                f"img_is_valid batch mismatch: expected {batch_size}, got {img_is_valid.shape[0]}"
            )
        if img_is_valid.shape[1] < seq_len:
            raise ValueError("img_is_valid must provide at least one flag per history step.")
        return img_is_valid[:, :seq_len].to(device=device).bool()

    def _apply_aug_transform(
        self,
        bev_seq: torch.Tensor,
        aug_transform: Any,
    ) -> torch.Tensor:
        if aug_transform is None:
            return bev_seq
        if isinstance(aug_transform, dict):
            flip_x = bool(aug_transform.get("flip_x", False))
            flip_y = bool(aug_transform.get("flip_y", False))
            if flip_x:
                bev_seq = torch.flip(bev_seq, dims=[-1])
            if flip_y:
                bev_seq = torch.flip(bev_seq, dims=[-2])
            return bev_seq
        if torch.is_tensor(aug_transform) and aug_transform.dim() >= 2:
            # Heuristic parity with BEV flip augmentation from upstream pipelines.
            if aug_transform.shape[-1] >= 2 and aug_transform.shape[-2] >= 2:
                diag = torch.diagonal(aug_transform[..., :2, :2], dim1=-2, dim2=-1)
                if torch.any(diag[..., 0] < 0):
                    bev_seq = torch.flip(bev_seq, dims=[-1])
                if torch.any(diag[..., 1] < 0):
                    bev_seq = torch.flip(bev_seq, dims=[-2])
        return bev_seq

    def _align_history_to_present(
        self,
        bev_seq: torch.Tensor,
        future_egomotion: torch.Tensor | None,
    ) -> torch.Tensor:
        if future_egomotion is None:
            return bev_seq

        aligned = bev_seq.clone()
        motion_xy = future_egomotion[..., :2]
        cumulative_motion = torch.flip(torch.cumsum(torch.flip(motion_xy, dims=[1]), dim=1), dims=[1])
        for batch_idx in range(aligned.shape[0]):
            for time_idx in range(aligned.shape[1]):
                shift_x = int(torch.round(cumulative_motion[batch_idx, time_idx, 0]).item())
                shift_y = int(torch.round(cumulative_motion[batch_idx, time_idx, 1]).item())
                if shift_x == 0 and shift_y == 0:
                    continue
                aligned[batch_idx, time_idx] = torch.roll(
                    aligned[batch_idx, time_idx],
                    shifts=(shift_y, shift_x),
                    dims=(-2, -1),
                )
        return aligned

    def _impute_invalid_history(
        self,
        bev_seq: torch.Tensor,
        img_is_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        if img_is_valid is None:
            return bev_seq
        hist_len = min(self.receptive_field, bev_seq.shape[1])
        x_valid = img_is_valid[:, :hist_len]
        filled = bev_seq.clone()
        for batch_idx in range(filled.shape[0]):
            if bool(x_valid[batch_idx].all()):
                continue
            invalid_positions = torch.where(~x_valid[batch_idx])[0]
            if invalid_positions.numel() == 0:
                continue
            invalid_index = int(invalid_positions[0].item())
            source_index = min(invalid_index + 1, hist_len - 1)
            valid_feat = filled[batch_idx, source_index].clone()
            filled[batch_idx, : invalid_index + 1] = valid_feat
        return filled

    def _initial_hidden_state(
        self,
        bev_seq: torch.Tensor,
        img_is_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        hist_len = min(self.receptive_field, bev_seq.shape[1])
        history = bev_seq[:, :hist_len]
        pooled = history.mean(dim=(-2, -1))
        if img_is_valid is not None:
            history_valid = img_is_valid[:, :hist_len].to(dtype=pooled.dtype).unsqueeze(-1)
            history_sum = (pooled * history_valid).sum(dim=1)
            denom = history_valid.sum(dim=1).clamp_min(1.0)
            history_context = history_sum / denom
        else:
            history_context = pooled.mean(dim=1)
        present_context = pooled[:, hist_len - 1]
        hidden = torch.tanh(self.init_proj(present_context) + self.history_proj(history_context))
        return hidden.unsqueeze(0)

    def forward(
        self,
        bev_embed: torch.Tensor,
        future_egomotion: torch.Tensor | None = None,
        aug_transform: Any = None,
        img_is_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bev_seq = self._as_sequence(bev_embed)
        batch_size, seq_len = bev_seq.shape[:2]
        future_egomotion = self._validate_future_egomotion(
            future_egomotion, batch_size, seq_len, bev_seq.device
        )
        img_is_valid = self._validate_img_is_valid(img_is_valid, batch_size, seq_len, bev_seq.device)

        bev_seq = self._apply_aug_transform(bev_seq, aug_transform=aug_transform)
        bev_seq = self._align_history_to_present(bev_seq, future_egomotion=future_egomotion)
        bev_seq = self._impute_invalid_history(bev_seq, img_is_valid=img_is_valid)

        h0 = self._initial_hidden_state(bev_seq, img_is_valid=img_is_valid)
        time_ids = torch.arange(self.cfg.pred_horizon, device=bev_seq.device).unsqueeze(0)
        queries = self.time_embedding(time_ids.expand(batch_size, -1))
        temporal_tokens, _ = self.gru(queries, h0)
        return temporal_tokens
