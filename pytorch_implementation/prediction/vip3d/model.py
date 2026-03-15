"""Pure-PyTorch VIP3D forward path with stricter upstream parity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from ..common.time_contracts import ensure_strictly_increasing as _shared_ensure_strictly_increasing
from .config import VIP3DConfig


def _masked_temporal_mean(sequence: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Compute masked mean over temporal dimension."""

    weights = valid_mask.to(sequence.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return (sequence * weights).sum(dim=1) / denom


def _last_observed_xy(agent_history: torch.Tensor, agent_valid: torch.Tensor) -> torch.Tensor:
    """Extract the last valid (x, y) position for each agent."""

    batch_size, num_agents, history_steps, _ = agent_history.shape
    time_index = torch.arange(history_steps, device=agent_history.device).view(1, 1, history_steps)
    masked_index = torch.where(agent_valid, time_index, torch.full_like(time_index, -1))
    last_index = masked_index.max(dim=-1).values.clamp(min=0)
    gather_index = last_index.unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_agents, 1, 2)
    return agent_history[..., :2].gather(dim=2, index=gather_index).squeeze(2)


def _as_batch_vector(
    value: Any,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """Normalize scalar or batch vector metadata into [B]."""

    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == 0:
        return tensor.repeat(batch_size)
    if tensor.ndim == 1 and tensor.shape[0] == batch_size:
        return tensor
    raise ValueError(f"{name} must be a scalar or tensor of shape [B], got {tuple(tensor.shape)}.")


def _ensure_strictly_increasing(indices: torch.Tensor, *, name: str) -> None:
    """Check strict monotonicity along the last dimension."""
    _shared_ensure_strictly_increasing(indices, name=name)


@dataclass
class VIP3DRuntimeState:
    """Stateful temporal cache mirroring ViP3D memory/query lifecycle."""

    query_tokens: torch.Tensor
    mem_bank: torch.Tensor
    mem_padding_mask: torch.Tensor
    save_period: torch.Tensor
    last_timestamp: torch.Tensor | None = None
    last_frame_index: torch.Tensor | None = None
    l2g_r_mat: torch.Tensor | None = None
    l2g_t: torch.Tensor | None = None


class DETRTrack3DCoderLite:
    """Small pure-PyTorch bbox coder mirroring ViP3D decode contracts."""

    def __init__(
        self,
        *,
        pc_range: tuple[float, float, float, float, float, float],
        post_center_range: tuple[float, float, float, float, float, float] | None,
        max_num: int,
        score_threshold: float | None,
        num_classes: int,
    ) -> None:
        self.pc_range = pc_range
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def _denormalize_bbox(self, bbox_preds: torch.Tensor) -> torch.Tensor:
        out = bbox_preds.clone()
        out[..., 0:1] = out[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        out[..., 1:2] = out[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        out[..., 4:5] = out[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        return out

    def decode_single(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        track_scores: torch.Tensor,
        obj_idxes: torch.Tensor,
        output_embedding: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if cls_scores.ndim != 2:
            raise ValueError("cls_scores must have shape [N, C].")
        if bbox_preds.ndim != 2:
            raise ValueError("bbox_preds must have shape [N, D].")

        max_num = min(self.max_num, cls_scores.shape[0])
        cls_probs = cls_scores.sigmoid()
        labels = cls_probs.argmax(dim=-1) % self.num_classes
        _, topk_idx = track_scores.topk(max_num)

        labels = labels[topk_idx]
        bbox_preds = bbox_preds[topk_idx]
        track_scores = track_scores[topk_idx]
        obj_idxes = obj_idxes[topk_idx]
        output_embedding = output_embedding[topk_idx]

        decoded_boxes = self._denormalize_bbox(bbox_preds)
        scores = track_scores

        if self.score_threshold is not None:
            score_mask = scores > self.score_threshold
        else:
            score_mask = torch.ones_like(scores, dtype=torch.bool)

        if self.post_center_range is not None:
            post = torch.tensor(self.post_center_range, device=scores.device, dtype=decoded_boxes.dtype)
            center_mask = (decoded_boxes[..., :3] >= post[:3]).all(dim=-1)
            center_mask &= (decoded_boxes[..., :3] <= post[3:]).all(dim=-1)
            mask = center_mask & score_mask
        else:
            mask = score_mask

        return {
            "bboxes": decoded_boxes[mask],
            "scores": scores[mask],
            "labels": labels[mask],
            "track_scores": track_scores[mask],
            "obj_idxes": obj_idxes[mask],
            "output_embedding": output_embedding[mask],
        }


class TemporalMemoryBankLite(nn.Module):
    """ViP3D-style temporal memory attention and gated memory updates."""

    def __init__(self, cfg: VIP3DConfig) -> None:
        super().__init__()
        self.save_thresh = cfg.memory_bank_score_thresh
        self.save_period = cfg.memory_bank_save_period
        self.max_his_length = cfg.memory_bank_len
        self.hidden_dim = cfg.hidden_dim

        if self.max_his_length > 0:
            nheads = 8 if cfg.hidden_dim % 8 == 0 else cfg.num_heads
            self.save_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
            self.temporal_attn = nn.MultiheadAttention(cfg.hidden_dim, nheads, dropout=0.0)
            self.temporal_fc1 = nn.Linear(cfg.hidden_dim, cfg.interaction_hidden_dim)
            self.temporal_fc2 = nn.Linear(cfg.interaction_hidden_dim, cfg.hidden_dim)
            self.temporal_norm1 = nn.LayerNorm(cfg.hidden_dim)
            self.temporal_norm2 = nn.LayerNorm(cfg.hidden_dim)
        else:
            self.save_proj = nn.Identity()
            self.temporal_attn = None
            self.temporal_fc1 = None
            self.temporal_fc2 = None
            self.temporal_norm1 = None
            self.temporal_norm2 = None

    def _forward_temporal_attn(
        self,
        output_embedding: torch.Tensor,
        mem_bank: torch.Tensor,
        mem_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.max_his_length == 0 or output_embedding.numel() == 0:
            return output_embedding

        valid_idxes = ~mem_padding_mask[:, -1]
        if not valid_idxes.any():
            return output_embedding

        embed = output_embedding[valid_idxes]
        prev_embed = mem_bank[valid_idxes]
        key_padding_mask = mem_padding_mask[valid_idxes]
        assert self.temporal_attn is not None
        assert self.temporal_fc1 is not None
        assert self.temporal_fc2 is not None
        assert self.temporal_norm1 is not None
        assert self.temporal_norm2 is not None

        embed2 = self.temporal_attn(
            embed.unsqueeze(0),
            prev_embed.transpose(0, 1),
            prev_embed.transpose(0, 1),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0][0]
        embed = self.temporal_norm1(embed + embed2)
        embed2 = self.temporal_fc2(F.relu(self.temporal_fc1(embed)))
        embed = self.temporal_norm2(embed + embed2)

        output_embedding = output_embedding.clone()
        output_embedding[valid_idxes] = embed
        return output_embedding

    def update(
        self,
        output_embedding: torch.Tensor,
        scores: torch.Tensor,
        mem_bank: torch.Tensor,
        mem_padding_mask: torch.Tensor,
        save_period: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.max_his_length == 0:
            return mem_bank, mem_padding_mask, save_period

        if self.training:
            saved_idxes = scores > 0
            updated_save_period = save_period
        else:
            updated_save_period = save_period.clone()
            saved_idxes = (updated_save_period == 0) & (scores > self.save_thresh)
            countdown = updated_save_period > 0
            updated_save_period[countdown] = updated_save_period[countdown] - 1
            updated_save_period[saved_idxes] = float(self.save_period)

        if not saved_idxes.any():
            return mem_bank, mem_padding_mask, updated_save_period

        assert isinstance(self.save_proj, nn.Module)
        saved_embed = self.save_proj(output_embedding[saved_idxes]).unsqueeze(1)
        prev_embed = mem_bank[saved_idxes]

        mem_bank = mem_bank.clone()
        mem_padding_mask = mem_padding_mask.clone()
        mem_bank[saved_idxes] = torch.cat([prev_embed[:, 1:], saved_embed], dim=1)
        new_valid = torch.zeros((saved_embed.shape[0], 1), dtype=torch.bool, device=output_embedding.device)
        mem_padding_mask[saved_idxes] = torch.cat([mem_padding_mask[saved_idxes, 1:], new_valid], dim=1)
        return mem_bank, mem_padding_mask, updated_save_period

    def forward(
        self,
        output_embedding: torch.Tensor,
        scores: torch.Tensor,
        mem_bank: torch.Tensor,
        mem_padding_mask: torch.Tensor,
        save_period: torch.Tensor,
        *,
        update_bank: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_agents, hidden_dim = output_embedding.shape
        flat_n = batch_size * num_agents

        flat_embed = output_embedding.reshape(flat_n, hidden_dim)
        flat_scores = scores.reshape(flat_n)
        flat_mem = mem_bank.reshape(flat_n, self.max_his_length, hidden_dim)
        flat_mask = mem_padding_mask.reshape(flat_n, self.max_his_length)
        flat_period = save_period.reshape(flat_n)

        flat_embed = self._forward_temporal_attn(flat_embed, flat_mem, flat_mask)
        if update_bank:
            flat_mem, flat_mask, flat_period = self.update(flat_embed, flat_scores, flat_mem, flat_mask, flat_period)

        return (
            flat_embed.view(batch_size, num_agents, hidden_dim),
            flat_mem.view(batch_size, num_agents, self.max_his_length, hidden_dim),
            flat_mask.view(batch_size, num_agents, self.max_his_length),
            flat_period.view(batch_size, num_agents),
        )


class QueryInteractionModuleLite(nn.Module):
    """Interaction-aware query update module adapted from ViP3D QIM."""

    def __init__(self, cfg: VIP3DConfig) -> None:
        super().__init__()
        self.random_drop = cfg.interaction_random_drop
        self.fp_ratio = cfg.interaction_fp_ratio
        self.update_query_pos = cfg.interaction_update_query_pos
        dropout = cfg.interaction_dropout

        nheads = 8 if cfg.hidden_dim % 8 == 0 else cfg.num_heads
        self.self_attn = nn.MultiheadAttention(cfg.hidden_dim, nheads, dropout)
        self.linear1 = nn.Linear(cfg.hidden_dim, cfg.interaction_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(cfg.interaction_hidden_dim, cfg.hidden_dim)

        if self.update_query_pos:
            self.linear_pos1 = nn.Linear(cfg.hidden_dim, cfg.interaction_hidden_dim)
            self.linear_pos2 = nn.Linear(cfg.interaction_hidden_dim, cfg.hidden_dim)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(cfg.hidden_dim)
        else:
            self.linear_pos1 = None
            self.linear_pos2 = None
            self.dropout_pos1 = None
            self.dropout_pos2 = None
            self.norm_pos = None

        self.linear_feat1 = nn.Linear(cfg.hidden_dim, cfg.interaction_hidden_dim)
        self.linear_feat2 = nn.Linear(cfg.interaction_hidden_dim, cfg.hidden_dim)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(cfg.hidden_dim)

        self.norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.norm2 = nn.LayerNorm(cfg.hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _augment_active_with_fp(
        self,
        active_idx: torch.Tensor,
        track_scores: torch.Tensor,
    ) -> torch.Tensor:
        if not self.training or self.fp_ratio <= 0:
            return active_idx

        num_active = int(active_idx.sum().item())
        if num_active == 0:
            return active_idx

        inactive_idx = ~active_idx
        num_inactive = int(inactive_idx.sum().item())
        if num_inactive == 0:
            return active_idx

        num_fp = min(int(round(self.fp_ratio * num_active)), num_inactive)
        if num_fp == 0:
            return active_idx

        inactive_positions = torch.nonzero(inactive_idx, as_tuple=False).squeeze(-1)
        inactive_scores = track_scores[inactive_positions]
        fp_pos = inactive_positions[torch.argsort(inactive_scores)[-num_fp:]]
        active_idx = active_idx.clone()
        active_idx[fp_pos] = True
        return active_idx

    def _random_drop_tracks(self, active_idx: torch.Tensor) -> torch.Tensor:
        if not self.training or self.random_drop <= 0 or not active_idx.any():
            return active_idx
        active_positions = torch.nonzero(active_idx, as_tuple=False).squeeze(-1)
        keep = torch.rand(active_positions.shape[0], device=active_idx.device) > self.random_drop
        dropped = torch.zeros_like(active_idx)
        dropped[active_positions[keep]] = True
        return dropped

    def forward(
        self,
        query_tokens: torch.Tensor,
        output_embedding: torch.Tensor,
        active_mask: torch.Tensor,
        track_scores: torch.Tensor,
    ) -> torch.Tensor:
        if query_tokens.numel() == 0:
            return query_tokens

        batch_size, num_agents, query_dim = query_tokens.shape
        hidden_dim = query_dim // 2
        updated_queries = query_tokens.clone()

        for b in range(batch_size):
            active_idx = active_mask[b]
            active_idx = self._random_drop_tracks(active_idx)
            active_idx = self._augment_active_with_fp(active_idx, track_scores[b])
            if not active_idx.any():
                continue

            track_query = updated_queries[b, active_idx]
            query_pos = track_query[:, :hidden_dim]
            query_feat = track_query[:, hidden_dim:]
            out_embed = output_embedding[b, active_idx]

            q = k = query_pos + out_embed
            tgt = out_embed
            tgt2 = self.self_attn(
                q[:, None, :],
                k[:, None, :],
                value=tgt[:, None, :],
                need_weights=False,
            )[0][:, 0, :]
            tgt = self.norm1(tgt + self.dropout1(tgt2))

            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
            tgt = self.norm2(tgt + self.dropout2(tgt2))

            if self.update_query_pos:
                assert self.linear_pos1 is not None
                assert self.linear_pos2 is not None
                assert self.dropout_pos1 is not None
                assert self.dropout_pos2 is not None
                assert self.norm_pos is not None
                query_pos2 = self.linear_pos2(self.dropout_pos1(F.relu(self.linear_pos1(tgt))))
                query_pos = self.norm_pos(query_pos + self.dropout_pos2(query_pos2))

            query_feat2 = self.linear_feat2(self.dropout_feat1(F.relu(self.linear_feat1(tgt))))
            query_feat = self.norm_feat(query_feat + self.dropout_feat2(query_feat2))
            updated_queries[b, active_idx] = torch.cat([query_pos, query_feat], dim=-1)

        return updated_queries


class TrajectoryDecoder(nn.Module):
    """Decode mode-wise trajectories with predictor-style probability semantics."""

    def __init__(self, cfg: VIP3DConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.trunk = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.decoder_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.decoder_hidden_dim, cfg.decoder_hidden_dim),
            nn.GELU(),
        )
        self.mode_head = nn.Linear(cfg.decoder_hidden_dim, cfg.num_modes)
        self.delta_head = nn.Linear(cfg.decoder_hidden_dim, cfg.num_modes * cfg.future_steps * 2)
        self.relative_decode = cfg.trajectory_relative

    def forward(self, fused_tokens: torch.Tensor, current_xy: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, num_agents, _ = fused_tokens.shape
        decoded = self.trunk(fused_tokens)
        mode_logits = self.mode_head(decoded)
        mode_log_probs = F.log_softmax(mode_logits, dim=-1)
        mode_probs = mode_log_probs.exp()

        modewise_raw = torch.cat([self.delta_head(decoded), mode_logits], dim=-1)
        raw_traj = modewise_raw[..., :-self.cfg.num_modes].view(
            batch_size,
            num_agents,
            self.cfg.num_modes,
            self.cfg.future_steps,
            2,
        )

        if self.relative_decode:
            trajectories = current_xy.unsqueeze(2).unsqueeze(3) + raw_traj.cumsum(dim=3)
        else:
            trajectories = raw_traj

        first_anchor = current_xy.unsqueeze(2).unsqueeze(3).expand(
            batch_size, num_agents, self.cfg.num_modes, 1, 2
        )
        prev_points = torch.cat([first_anchor, trajectories[:, :, :, :-1, :]], dim=3)
        traj_deltas = trajectories - prev_points

        best_mode = mode_probs.argmax(dim=-1)
        gather_idx = best_mode.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
            batch_size, num_agents, 1, self.cfg.future_steps, 2
        )
        best_trajectory = trajectories.gather(dim=2, index=gather_idx).squeeze(2)

        return {
            "mode_logits": mode_logits,
            "mode_log_probs": mode_log_probs,
            "mode_probs": mode_probs,
            "traj_deltas": traj_deltas,
            "trajectories": trajectories,
            "best_mode": best_mode,
            "best_trajectory": best_trajectory,
            "pred_outputs": trajectories,
            "pred_probs": mode_probs,
        }


class VIP3DLite(nn.Module):
    """Educational forward path with ViP3D-like temporal/query behavior."""

    def __init__(self, cfg: VIP3DConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.history_input_proj = nn.Linear(cfg.agent_input_dim, cfg.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 2,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.encoder_layers)
        self.history_norm = nn.LayerNorm(cfg.hidden_dim)

        self.map_point_mlp = nn.Sequential(
            nn.Linear(cfg.map_input_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.map_token_proj = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

        self.agent_map_attention = nn.MultiheadAttention(
            embed_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.fusion_norm = nn.LayerNorm(cfg.hidden_dim)

        self.track_score_head = nn.Linear(cfg.hidden_dim, 1)
        self.memory_bank = TemporalMemoryBankLite(cfg)
        self.query_interaction = QueryInteractionModuleLite(cfg)
        self.decoder = TrajectoryDecoder(cfg)
        self.bbox_coder = DETRTrack3DCoderLite(
            pc_range=cfg.pc_range,
            post_center_range=cfg.post_center_range,
            max_num=cfg.decode_max_num,
            score_threshold=cfg.decode_score_threshold,
            num_classes=cfg.num_modes,
        )

    def _build_runtime_state(
        self,
        *,
        batch_size: int,
        num_agents: int,
        seed_tokens: torch.Tensor,
    ) -> VIP3DRuntimeState:
        query_tokens = torch.cat([seed_tokens, seed_tokens], dim=-1).clone()
        mem_bank = seed_tokens.new_zeros(batch_size, num_agents, self.cfg.memory_bank_len, self.cfg.hidden_dim)
        mem_padding_mask = torch.ones(batch_size, num_agents, self.cfg.memory_bank_len, device=seed_tokens.device, dtype=torch.bool)
        save_period = seed_tokens.new_zeros(batch_size, num_agents)
        return VIP3DRuntimeState(
            query_tokens=query_tokens,
            mem_bank=mem_bank,
            mem_padding_mask=mem_padding_mask,
            save_period=save_period,
        )

    def _coerce_runtime_state(
        self,
        runtime_state: VIP3DRuntimeState | Mapping[str, torch.Tensor] | None,
        *,
        batch_size: int,
        num_agents: int,
        seed_tokens: torch.Tensor,
    ) -> VIP3DRuntimeState:
        if runtime_state is None:
            return self._build_runtime_state(batch_size=batch_size, num_agents=num_agents, seed_tokens=seed_tokens)

        if isinstance(runtime_state, Mapping):
            query_tokens = runtime_state["query_tokens"]
            mem_bank = runtime_state["mem_bank"]
            mem_padding_mask = runtime_state["mem_padding_mask"]
            save_period = runtime_state["save_period"]
            state = VIP3DRuntimeState(
                query_tokens=query_tokens,
                mem_bank=mem_bank,
                mem_padding_mask=mem_padding_mask,
                save_period=save_period,
                last_timestamp=runtime_state.get("last_timestamp"),
                last_frame_index=runtime_state.get("last_frame_index"),
                l2g_r_mat=runtime_state.get("l2g_r_mat"),
                l2g_t=runtime_state.get("l2g_t"),
            )
        else:
            state = runtime_state

        hidden_dim = self.cfg.hidden_dim
        expected_query_shape = (batch_size, num_agents, hidden_dim * 2)
        expected_mem_shape = (batch_size, num_agents, self.cfg.memory_bank_len, hidden_dim)
        expected_mask_shape = (batch_size, num_agents, self.cfg.memory_bank_len)
        expected_period_shape = (batch_size, num_agents)

        if tuple(state.query_tokens.shape) != expected_query_shape:
            raise ValueError(f"runtime_state.query_tokens must be {expected_query_shape}, got {tuple(state.query_tokens.shape)}.")
        if tuple(state.mem_bank.shape) != expected_mem_shape:
            raise ValueError(f"runtime_state.mem_bank must be {expected_mem_shape}, got {tuple(state.mem_bank.shape)}.")
        if tuple(state.mem_padding_mask.shape) != expected_mask_shape:
            raise ValueError(
                f"runtime_state.mem_padding_mask must be {expected_mask_shape}, got {tuple(state.mem_padding_mask.shape)}."
            )
        if tuple(state.save_period.shape) != expected_period_shape:
            raise ValueError(f"runtime_state.save_period must be {expected_period_shape}, got {tuple(state.save_period.shape)}.")

        return VIP3DRuntimeState(
            query_tokens=state.query_tokens.to(device=seed_tokens.device, dtype=seed_tokens.dtype),
            mem_bank=state.mem_bank.to(device=seed_tokens.device, dtype=seed_tokens.dtype),
            mem_padding_mask=state.mem_padding_mask.to(device=seed_tokens.device, dtype=torch.bool),
            save_period=state.save_period.to(device=seed_tokens.device, dtype=seed_tokens.dtype),
            last_timestamp=None if state.last_timestamp is None else state.last_timestamp.to(device=seed_tokens.device, dtype=seed_tokens.dtype),
            last_frame_index=None if state.last_frame_index is None else state.last_frame_index.to(device=seed_tokens.device, dtype=torch.long),
            l2g_r_mat=None if state.l2g_r_mat is None else state.l2g_r_mat.to(device=seed_tokens.device, dtype=seed_tokens.dtype),
            l2g_t=None if state.l2g_t is None else state.l2g_t.to(device=seed_tokens.device, dtype=seed_tokens.dtype),
        )

    def _validate_metadata_contract(
        self,
        metadata: Mapping[str, Any] | None,
        *,
        batch_size: int,
        history_steps: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor | None]:
        default_history_indices = torch.arange(history_steps, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        contract: dict[str, torch.Tensor | None] = {
            "history_time_indices": default_history_indices,
            "timestamp": None,
            "frame_index": None,
            "l2g_r_mat": None,
            "l2g_t": None,
        }

        if metadata is None:
            return contract
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a mapping when provided.")

        if "history_time_indices" in metadata:
            history_indices = torch.as_tensor(metadata["history_time_indices"], device=device, dtype=torch.long)
            if history_indices.ndim == 1:
                if history_indices.shape[0] != history_steps:
                    raise ValueError(
                        f"metadata['history_time_indices'] must have length {history_steps}, got {history_indices.shape[0]}."
                    )
                history_indices = history_indices.unsqueeze(0).expand(batch_size, -1)
            elif history_indices.ndim == 2 and history_indices.shape == (batch_size, history_steps):
                pass
            else:
                raise ValueError(
                    "metadata['history_time_indices'] must have shape [T_h] or [B, T_h] "
                    f"with [B, T_h]={[batch_size, history_steps]}."
                )
            _ensure_strictly_increasing(history_indices, name="metadata['history_time_indices']")
            contract["history_time_indices"] = history_indices

        if "timestamp" in metadata:
            contract["timestamp"] = _as_batch_vector(
                metadata["timestamp"],
                batch_size=batch_size,
                device=device,
                dtype=torch.float32,
                name="metadata['timestamp']",
            )
        elif "timestamps" in metadata:
            contract["timestamp"] = _as_batch_vector(
                metadata["timestamps"],
                batch_size=batch_size,
                device=device,
                dtype=torch.float32,
                name="metadata['timestamps']",
            )

        if "frame_index" in metadata:
            contract["frame_index"] = _as_batch_vector(
                metadata["frame_index"],
                batch_size=batch_size,
                device=device,
                dtype=torch.long,
                name="metadata['frame_index']",
            )
        elif "index" in metadata:
            contract["frame_index"] = _as_batch_vector(
                metadata["index"],
                batch_size=batch_size,
                device=device,
                dtype=torch.long,
                name="metadata['index']",
            )

        if "l2g_r_mat" in metadata:
            l2g_r_mat = torch.as_tensor(metadata["l2g_r_mat"], device=device, dtype=torch.float32)
            if l2g_r_mat.ndim == 2:
                l2g_r_mat = l2g_r_mat.unsqueeze(0).expand(batch_size, -1, -1)
            elif l2g_r_mat.ndim == 3 and l2g_r_mat.shape[0] == batch_size:
                pass
            else:
                raise ValueError("metadata['l2g_r_mat'] must have shape [3, 3] or [B, 3, 3].")
            if tuple(l2g_r_mat.shape[1:]) != (3, 3):
                raise ValueError("metadata['l2g_r_mat'] must end with shape [3, 3].")
            contract["l2g_r_mat"] = l2g_r_mat

        if "l2g_t" in metadata:
            l2g_t = torch.as_tensor(metadata["l2g_t"], device=device, dtype=torch.float32)
            if l2g_t.ndim == 1:
                if l2g_t.shape[0] != 3:
                    raise ValueError("metadata['l2g_t'] must have shape [3], [B, 3], or [B, 1, 3].")
                l2g_t = l2g_t.unsqueeze(0).expand(batch_size, -1)
            elif l2g_t.ndim == 2 and l2g_t.shape == (batch_size, 3):
                pass
            elif l2g_t.ndim == 3 and l2g_t.shape == (batch_size, 1, 3):
                l2g_t = l2g_t.squeeze(1)
            else:
                raise ValueError("metadata['l2g_t'] must have shape [3], [B, 3], or [B, 1, 3].")
            contract["l2g_t"] = l2g_t

        return contract

    def _apply_temporal_contract(
        self,
        state: VIP3DRuntimeState,
        contract: dict[str, torch.Tensor | None],
        *,
        seed_tokens: torch.Tensor,
        runtime_state_provided: bool,
    ) -> tuple[VIP3DRuntimeState, dict[str, torch.Tensor]]:
        batch_size, num_agents, _ = seed_tokens.shape
        dtype = seed_tokens.dtype
        device = seed_tokens.device

        timestamp = contract["timestamp"]
        frame_index = contract["frame_index"]
        time_delta = torch.zeros(batch_size, device=device, dtype=dtype)
        frame_delta = torch.zeros(batch_size, device=device, dtype=torch.long)
        reset_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)

        if runtime_state_provided and self.cfg.enforce_strict_metadata and timestamp is None and frame_index is None:
            raise ValueError("metadata must include 'timestamp' or 'frame_index' when runtime_state is reused.")

        if timestamp is not None and state.last_timestamp is not None:
            time_delta = timestamp - state.last_timestamp
            if (time_delta < 0).any():
                raise ValueError("metadata timestamp must be non-decreasing across calls.")
            reset_mask = time_delta > self.cfg.metadata_time_gap_reset

        if frame_index is not None and state.last_frame_index is not None:
            frame_delta = frame_index - state.last_frame_index
            if (frame_delta < 0).any():
                raise ValueError("metadata frame_index must be non-decreasing across calls.")
            if timestamp is None:
                reset_mask = reset_mask | (frame_delta > 1)

        if reset_mask.any():
            reset_idx = torch.nonzero(reset_mask, as_tuple=False).squeeze(-1)
            state.query_tokens = state.query_tokens.clone()
            state.query_tokens[reset_idx] = torch.cat([seed_tokens[reset_idx], seed_tokens[reset_idx]], dim=-1)
            state.mem_bank = state.mem_bank.clone()
            state.mem_bank[reset_idx] = 0
            state.mem_padding_mask = state.mem_padding_mask.clone()
            state.mem_padding_mask[reset_idx] = True
            state.save_period = state.save_period.clone()
            state.save_period[reset_idx] = 0

        state.last_timestamp = timestamp if timestamp is not None else state.last_timestamp
        state.last_frame_index = frame_index if frame_index is not None else state.last_frame_index
        state.l2g_r_mat = contract["l2g_r_mat"] if contract["l2g_r_mat"] is not None else state.l2g_r_mat
        state.l2g_t = contract["l2g_t"] if contract["l2g_t"] is not None else state.l2g_t

        history_time_indices = contract["history_time_indices"]
        assert history_time_indices is not None
        timestamp_out = timestamp if timestamp is not None else torch.zeros(batch_size, device=device, dtype=dtype)
        frame_index_out = frame_index if frame_index is not None else torch.zeros(batch_size, device=device, dtype=torch.long)

        return state, {
            "history_time_indices": history_time_indices,
            "timestamp": timestamp_out,
            "frame_index": frame_index_out,
            "time_delta": time_delta,
            "frame_delta": frame_delta,
            "reset_mask": reset_mask,
        }

    def forward(
        self,
        agent_history: torch.Tensor,
        map_polylines: torch.Tensor,
        agent_valid: torch.Tensor | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
        runtime_state: VIP3DRuntimeState | Mapping[str, torch.Tensor] | None = None,
        update_memory: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Run ViP3D-style multimodal trajectory prediction.

        Args:
            agent_history: [B, A, T_h, F_a] motion history.
            map_polylines: [B, P, S, F_m] map token points.
            agent_valid: [B, A, T_h] valid history mask.
            metadata: Optional per-frame metadata for strict time contracts.
            runtime_state: Optional memory/query state carried across frames.
            update_memory: Whether memory bank writes current embeddings.
        """

        if agent_history.ndim != 4:
            raise ValueError("agent_history must have shape [B, A, T_h, F_a].")
        if map_polylines.ndim != 4:
            raise ValueError("map_polylines must have shape [B, P, S, F_m].")

        batch_size, num_agents, history_steps, feat_dim = agent_history.shape
        _, num_map_tokens, points_per_token, map_dim = map_polylines.shape
        if feat_dim != self.cfg.agent_input_dim:
            raise ValueError(f"Expected agent_input_dim={self.cfg.agent_input_dim}, got {feat_dim}.")
        if map_dim != self.cfg.map_input_dim:
            raise ValueError(f"Expected map_input_dim={self.cfg.map_input_dim}, got {map_dim}.")
        if history_steps != self.cfg.history_steps:
            raise ValueError(f"Expected history_steps={self.cfg.history_steps}, got {history_steps}.")
        if num_map_tokens != self.cfg.map_tokens:
            raise ValueError(f"Expected map_tokens={self.cfg.map_tokens}, got {num_map_tokens}.")
        if points_per_token != self.cfg.map_points_per_token:
            raise ValueError(
                f"Expected map_points_per_token={self.cfg.map_points_per_token}, got {points_per_token}."
            )

        if agent_valid is None:
            agent_valid = torch.ones(
                batch_size,
                num_agents,
                history_steps,
                device=agent_history.device,
                dtype=torch.bool,
            )
        else:
            expected_valid_shape = (batch_size, num_agents, history_steps)
            if tuple(agent_valid.shape) != expected_valid_shape:
                raise ValueError(f"agent_valid must have shape {expected_valid_shape}, got {tuple(agent_valid.shape)}.")
            agent_valid = agent_valid.to(device=agent_history.device, dtype=torch.bool)

        metadata_contract = self._validate_metadata_contract(
            metadata,
            batch_size=batch_size,
            history_steps=history_steps,
            device=agent_history.device,
        )

        history_flat = agent_history.view(batch_size * num_agents, history_steps, feat_dim)
        valid_flat = agent_valid.view(batch_size * num_agents, history_steps)
        empty_rows = ~valid_flat.any(dim=1)
        if empty_rows.any():
            valid_flat = valid_flat.clone()
            valid_flat[empty_rows, 0] = True

        history_tokens = self.history_input_proj(history_flat)
        history_encoded = self.history_encoder(history_tokens, src_key_padding_mask=~valid_flat)
        history_encoded = self.history_norm(history_encoded)
        history_tokens_batched = history_encoded.view(batch_size, num_agents, history_steps, self.cfg.hidden_dim)
        agent_tokens = _masked_temporal_mean(history_encoded, valid_flat).view(batch_size, num_agents, -1)

        map_points = self.map_point_mlp(map_polylines)
        map_tokens = self.map_token_proj(map_points.mean(dim=2))
        fused_tokens, attention_weights = self.agent_map_attention(
            query=agent_tokens,
            key=map_tokens,
            value=map_tokens,
            need_weights=True,
        )
        fused_tokens = self.fusion_norm(agent_tokens + fused_tokens)

        runtime_state_obj = self._coerce_runtime_state(
            runtime_state,
            batch_size=batch_size,
            num_agents=num_agents,
            seed_tokens=fused_tokens,
        )
        runtime_state_obj, temporal_contract = self._apply_temporal_contract(
            runtime_state_obj,
            metadata_contract,
            seed_tokens=fused_tokens,
            runtime_state_provided=runtime_state is not None,
        )

        current_xy = _last_observed_xy(agent_history, agent_valid)
        track_scores = torch.sigmoid(self.track_score_head(fused_tokens)).squeeze(-1).clamp(min=1e-6, max=1 - 1e-6)
        memory_tokens, mem_bank, mem_padding_mask, save_period = self.memory_bank(
            fused_tokens,
            track_scores,
            runtime_state_obj.mem_bank,
            runtime_state_obj.mem_padding_mask,
            runtime_state_obj.save_period,
            update_bank=update_memory,
        )

        active_mask = agent_valid.any(dim=-1)
        query_tokens = self.query_interaction(runtime_state_obj.query_tokens, memory_tokens, active_mask, track_scores)
        decoded = self.decoder(memory_tokens, current_xy)

        bbox_stub = torch.zeros(batch_size, num_agents, 10, device=memory_tokens.device, dtype=memory_tokens.dtype)
        bbox_stub[..., 0:2] = current_xy
        bbox_stub[..., -2:] = decoded["traj_deltas"][:, :, 0, 0, :]
        obj_idxes = torch.arange(num_agents, device=memory_tokens.device, dtype=torch.long)
        decoded_tracks = [
            self.bbox_coder.decode_single(
                cls_scores=decoded["mode_logits"][i],
                bbox_preds=bbox_stub[i],
                track_scores=track_scores[i],
                obj_idxes=obj_idxes,
                output_embedding=memory_tokens[i],
            )
            for i in range(batch_size)
        ]

        state_out = {
            "query_tokens": query_tokens.detach(),
            "mem_bank": mem_bank.detach(),
            "mem_padding_mask": mem_padding_mask.detach(),
            "save_period": save_period.detach(),
            "last_timestamp": temporal_contract["timestamp"].detach(),
            "last_frame_index": temporal_contract["frame_index"].detach(),
            "l2g_r_mat": (
                runtime_state_obj.l2g_r_mat.detach()
                if runtime_state_obj.l2g_r_mat is not None
                else torch.eye(3, device=memory_tokens.device, dtype=memory_tokens.dtype).unsqueeze(0).expand(batch_size, -1, -1)
            ),
            "l2g_t": (
                runtime_state_obj.l2g_t.detach()
                if runtime_state_obj.l2g_t is not None
                else torch.zeros(batch_size, 3, device=memory_tokens.device, dtype=memory_tokens.dtype)
            ),
        }

        return {
            "history_tokens": history_tokens_batched,
            "map_tokens": map_tokens,
            "fused_tokens": fused_tokens,
            "attention_weights": attention_weights,
            "memory_tokens": memory_tokens,
            "query_tokens": query_tokens,
            "track_scores": track_scores,
            "history_time_indices": temporal_contract["history_time_indices"],
            "timestamp": temporal_contract["timestamp"],
            "frame_index": temporal_contract["frame_index"],
            "time_delta": temporal_contract["time_delta"],
            "frame_delta": temporal_contract["frame_delta"],
            "temporal_reset_mask": temporal_contract["reset_mask"],
            "decoded_tracks": decoded_tracks,
            "runtime_state": state_out,
            **decoded,
        }


def compute_ade_fde(
    trajectories: torch.Tensor,
    gt_future: torch.Tensor,
    gt_valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute a simple best-of-K ADE/FDE for smoke testing."""

    if trajectories.ndim != 5:
        raise ValueError("trajectories must have shape [B, A, M, T_f, 2].")
    if gt_future.ndim != 4:
        raise ValueError("gt_future must have shape [B, A, T_f, 2].")

    batch_size, num_agents, _, horizon, _ = trajectories.shape
    if gt_future.shape[:3] != (batch_size, num_agents, horizon):
        raise ValueError("gt_future must match [B, A, T_f] of trajectories.")

    if gt_valid is None:
        gt_valid = torch.ones(batch_size, num_agents, horizon, dtype=torch.bool, device=trajectories.device)
    else:
        gt_valid = gt_valid.to(dtype=torch.bool)

    disp = torch.linalg.norm(trajectories - gt_future.unsqueeze(2), dim=-1)
    valid = gt_valid.unsqueeze(2).to(disp.dtype)
    ade_per_mode = (disp * valid).sum(dim=-1) / valid.sum(dim=-1).clamp(min=1.0)

    time_index = torch.arange(horizon, device=trajectories.device).view(1, 1, horizon)
    masked_index = torch.where(gt_valid, time_index, torch.full_like(time_index, -1))
    last_index = masked_index.max(dim=-1).values.clamp(min=0)
    gather_index = last_index.unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_agents, disp.shape[2], 1)
    fde_per_mode = disp.gather(dim=3, index=gather_index).squeeze(-1)

    best_mode = fde_per_mode.argmin(dim=-1)
    best_gather = best_mode.unsqueeze(-1)
    ade_best = ade_per_mode.gather(dim=2, index=best_gather).squeeze(-1)
    fde_best = fde_per_mode.gather(dim=2, index=best_gather).squeeze(-1)

    valid_agents = gt_valid.any(dim=-1)
    ade = ade_best[valid_agents].mean() if valid_agents.any() else trajectories.new_tensor(0.0)
    fde = fde_best[valid_agents].mean() if valid_agents.any() else trajectories.new_tensor(0.0)

    return {
        "ade": ade,
        "fde": fde,
        "best_mode": best_mode,
        "ade_per_agent": ade_best,
        "fde_per_agent": fde_best,
    }


def _collect_tensors(value: Any) -> list[torch.Tensor]:
    """Internal helper for debug/tests."""

    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (tuple, list)):
        tensors: list[torch.Tensor] = []
        for item in value:
            tensors.extend(_collect_tensors(item))
        return tensors
    if isinstance(value, dict):
        tensors = []
        for item in value.values():
            tensors.extend(_collect_tensors(item))
        return tensors
    return []
