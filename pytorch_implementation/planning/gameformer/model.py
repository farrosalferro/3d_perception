"""Pure-PyTorch GameFormer-style end-to-end planning forward path."""

from __future__ import annotations

import torch
from torch import nn

from ..common import (
    PLANNING_OUTPUT_KEYS,
    PlanningBatch,
    acceleration_from_velocity,
    build_time_stamps,
    curvature_from_trajectory,
    kinematic_feasible_mask,
    validate_planning_batch,
    velocity_from_trajectory,
)
from .config import GameFormerPlanningConfig


def _pairwise_min_distance(
    candidate_trajectories: torch.Tensor,
    agent_trajectories: torch.Tensor,
) -> torch.Tensor:
    # [B, K, T, 2] vs [B, A, T, 2] -> [B, K]
    diff = candidate_trajectories[:, :, None, :, :] - agent_trajectories[:, None, :, :, :]
    return torch.linalg.norm(diff, dim=-1).amin(dim=(-1, -2))


class GameFormerSceneEncoder(nn.Module):
    """Encode ego history, agents, map vectors, and optional route features."""

    def __init__(self, cfg: GameFormerPlanningConfig) -> None:
        super().__init__()
        hidden = cfg.e2e.hidden_dim
        self.ego_proj = nn.Linear(cfg.e2e.state_dim, hidden)
        self.ego_gru = nn.GRU(input_size=hidden, hidden_size=hidden, batch_first=True)
        self.agent_proj = nn.Linear(cfg.e2e.state_dim, hidden)
        self.map_point_proj = nn.Linear(cfg.e2e.map_feat_dim, hidden)
        self.route_proj = nn.Linear(4, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, batch: PlanningBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ego_tokens = self.ego_proj(batch.ego_history)
        _, ego_hidden = self.ego_gru(ego_tokens)
        ego_token = ego_hidden[-1].unsqueeze(1)  # [B, 1, C]
        agent_tokens = self.agent_proj(batch.agent_states)
        map_tokens = self.map_point_proj(batch.map_polylines).mean(dim=2)
        route_tokens = None
        if batch.route_features is not None:
            route_tokens = self.route_proj(batch.route_features)
        return self.norm(ego_token), self.norm(agent_tokens), self.norm(map_tokens if route_tokens is None else torch.cat((map_tokens, route_tokens), dim=1))


class GameFormerInteractionLayer(nn.Module):
    """One game-theoretic refinement step over ego and agent tokens."""

    def __init__(self, cfg: GameFormerPlanningConfig) -> None:
        super().__init__()
        hidden = cfg.e2e.hidden_dim
        self.ego_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.agent_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.ego_ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden * 2, hidden),
        )
        self.agent_ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden * 2, hidden),
        )
        self.ego_norm = nn.LayerNorm(hidden)
        self.agent_norm = nn.LayerNorm(hidden)

    def forward(
        self,
        ego_token: torch.Tensor,
        agent_tokens: torch.Tensor,
        map_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ego_context = torch.cat((agent_tokens, map_tokens), dim=1)
        ego_attended, _ = self.ego_attn(query=ego_token, key=ego_context, value=ego_context, need_weights=False)
        ego_token = self.ego_norm(ego_token + ego_attended + self.ego_ffn(ego_token))

        agent_context = torch.cat((ego_token, map_tokens), dim=1)
        agent_attended, _ = self.agent_attn(query=agent_tokens, key=agent_context, value=agent_context, need_weights=False)
        agent_tokens = self.agent_norm(agent_tokens + agent_attended + self.agent_ffn(agent_tokens))
        return ego_token, agent_tokens


class GameFormerLite(nn.Module):
    """Educational GameFormer-style planner under `planning/gameformer`."""

    def __init__(self, cfg: GameFormerPlanningConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.scene_encoder = GameFormerSceneEncoder(cfg)
        self.interaction_layers = nn.ModuleList([GameFormerInteractionLayer(cfg) for _ in range(cfg.game_levels)])
        hidden = cfg.e2e.hidden_dim
        k = cfg.e2e.num_candidates
        t = cfg.e2e.future_steps

        self.candidate_delta_head = nn.Linear(hidden, k * t * 2)
        self.candidate_score_head = nn.Linear(hidden, k)
        self.agent_delta_head = nn.Linear(hidden, t * 2)
        self.max_step = cfg.e2e.max_speed * cfg.e2e.dt
        self.interactive_gain = float(cfg.interactive_gain)

    def _decode_candidates(self, ego_token: torch.Tensor, start_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = ego_token.shape[0]
        k = self.cfg.e2e.num_candidates
        t = self.cfg.e2e.future_steps
        raw = self.candidate_delta_head(ego_token).view(batch_size, k, t, 2)
        deltas = torch.tanh(raw) * self.max_step
        trajectories = start_xy[:, None, None, :] + torch.cumsum(deltas, dim=2)
        logits = self.candidate_score_head(ego_token)
        return trajectories, deltas, logits

    def _decode_agent_futures(self, agent_tokens: torch.Tensor, agent_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_agents, _ = agent_tokens.shape
        t = self.cfg.e2e.future_steps
        raw = self.agent_delta_head(agent_tokens).view(batch_size, num_agents, t, 2)
        deltas = torch.tanh(raw) * (self.max_step * self.interactive_gain)
        start_xy = agent_states[..., :2].unsqueeze(2)
        linear = agent_states[..., 2:4].unsqueeze(2) * (
            torch.arange(1, t + 1, device=agent_tokens.device, dtype=agent_tokens.dtype).view(1, 1, t, 1)
            * self.cfg.e2e.dt
        )
        return start_xy + linear + torch.cumsum(deltas, dim=2)

    def forward(
        self,
        ego_history: torch.Tensor,
        agent_states: torch.Tensor,
        map_polylines: torch.Tensor,
        route_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = PlanningBatch(
            ego_history=ego_history,
            agent_states=agent_states,
            map_polylines=map_polylines,
            route_features=route_features,
        )
        validate_planning_batch(batch, self.cfg.e2e)

        ego_token, agent_tokens, map_tokens = self.scene_encoder(batch)
        level_ego_tokens = []
        level_agent_tokens = []
        for layer in self.interaction_layers:
            ego_token, agent_tokens = layer(ego_token, agent_tokens, map_tokens)
            level_ego_tokens.append(ego_token.squeeze(1))
            level_agent_tokens.append(agent_tokens)

        final_ego = ego_token.squeeze(1)
        start_xy = batch.ego_history[:, -1, :2]
        candidate_trajectories, deltas, candidate_logits = self._decode_candidates(final_ego, start_xy)
        candidate_scores = torch.softmax(candidate_logits, dim=-1)

        agent_future = self._decode_agent_futures(agent_tokens, batch.agent_states)

        velocity = velocity_from_trajectory(candidate_trajectories, dt=self.cfg.e2e.dt)
        acceleration = acceleration_from_velocity(velocity, dt=self.cfg.e2e.dt)
        curvature = curvature_from_trajectory(candidate_trajectories)
        feasible_mask = kinematic_feasible_mask(
            velocity=velocity,
            acceleration=acceleration,
            curvature=curvature,
            max_speed=self.cfg.e2e.max_speed,
            max_accel=self.cfg.e2e.max_accel,
            max_curvature=self.cfg.e2e.max_curvature,
        )

        min_distance = _pairwise_min_distance(candidate_trajectories, agent_future)
        collision_free = min_distance >= float(self.cfg.e2e.safety_margin)
        safety_margin_violation = torch.relu(self.cfg.e2e.safety_margin - min_distance)

        valid = feasible_mask & collision_free
        safe_scores = candidate_scores.masked_fill(~valid, -1.0)
        safe_best = safe_scores.argmax(dim=-1)
        fallback_best = candidate_scores.argmax(dim=-1)
        selected_index = torch.where(valid.any(dim=-1), safe_best, fallback_best)
        gather_idx = selected_index[:, None, None, None].expand(-1, 1, self.cfg.e2e.future_steps, 2)
        selected_trajectory = candidate_trajectories.gather(dim=1, index=gather_idx).squeeze(1)

        time_stamps = build_time_stamps(
            future_steps=self.cfg.e2e.future_steps,
            dt=self.cfg.e2e.dt,
            device=candidate_trajectories.device,
            dtype=candidate_trajectories.dtype,
        )
        outputs = {
            "level_ego_tokens": torch.stack(level_ego_tokens, dim=1),
            "level_agent_tokens": torch.stack(level_agent_tokens, dim=1),
            "candidate_logits": candidate_logits,
            "candidate_deltas": deltas,
            "candidate_trajectories": candidate_trajectories,
            "candidate_scores": candidate_scores,
            "selected_index": selected_index,
            "selected_trajectory": selected_trajectory,
            "velocity": velocity,
            "acceleration": acceleration,
            "curvature": curvature,
            "feasible_mask": feasible_mask,
            "collision_free_mask": collision_free,
            "safety_margin_violation": safety_margin_violation,
            "min_distance": min_distance,
            "agent_future": agent_future,
            "time_stamps": time_stamps,
        }
        for key in PLANNING_OUTPUT_KEYS:
            if key not in outputs:
                raise KeyError(f"Missing required planning output key: {key}")
        return outputs
