"""Pure PyTorch VIP3D-style trajectory prediction forward path."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

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


class TrajectoryDecoder(nn.Module):
    """Decode multimodal trajectory deltas and mode probabilities."""

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

    def forward(self, fused_tokens: torch.Tensor, current_xy: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, num_agents, _ = fused_tokens.shape
        decoded = self.trunk(fused_tokens)
        mode_logits = self.mode_head(decoded)
        mode_probs = mode_logits.softmax(dim=-1)

        deltas = self.delta_head(decoded).view(
            batch_size,
            num_agents,
            self.cfg.num_modes,
            self.cfg.future_steps,
            2,
        )
        trajectories = current_xy.unsqueeze(2).unsqueeze(3) + deltas.cumsum(dim=3)
        best_mode = mode_probs.argmax(dim=-1)
        gather_idx = best_mode.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
            batch_size, num_agents, 1, self.cfg.future_steps, 2
        )
        best_trajectory = trajectories.gather(dim=2, index=gather_idx).squeeze(2)

        return {
            "mode_logits": mode_logits,
            "mode_probs": mode_probs,
            "traj_deltas": deltas,
            "trajectories": trajectories,
            "best_mode": best_mode,
            "best_trajectory": best_trajectory,
        }


class VIP3DLite(nn.Module):
    """Educational forward-only trajectory predictor inspired by VIP3D."""

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

        self.decoder = TrajectoryDecoder(cfg)

    def forward(
        self,
        agent_history: torch.Tensor,
        map_polylines: torch.Tensor,
        agent_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run VIP3D-style multimodal trajectory prediction.

        Args:
            agent_history: [B, A, T_h, F_a] motion history.
            map_polylines: [B, P, S, 2] map token points.
            agent_valid: [B, A, T_h] valid history mask.
        """

        if agent_history.ndim != 4:
            raise ValueError("agent_history must have shape [B, A, T_h, F_a].")
        if map_polylines.ndim != 4:
            raise ValueError("map_polylines must have shape [B, P, S, 2].")

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
            agent_valid = agent_valid.to(device=agent_history.device, dtype=torch.bool)

        history_flat = agent_history.view(batch_size * num_agents, history_steps, feat_dim)
        valid_flat = agent_valid.view(batch_size * num_agents, history_steps)
        empty_rows = ~valid_flat.any(dim=1)
        if empty_rows.any():
            valid_flat = valid_flat.clone()
            valid_flat[empty_rows, 0] = True
        history_tokens = self.history_input_proj(history_flat)
        history_encoded = self.history_encoder(history_tokens, src_key_padding_mask=~valid_flat)
        history_encoded = self.history_norm(history_encoded)
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
        current_xy = _last_observed_xy(agent_history, agent_valid)
        decoded = self.decoder(fused_tokens, current_xy)

        return {
            "history_tokens": history_encoded.view(batch_size, num_agents, history_steps, self.cfg.hidden_dim),
            "map_tokens": map_tokens,
            "fused_tokens": fused_tokens,
            "attention_weights": attention_weights,
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
