"""Pure-PyTorch UniAD-style end-to-end planning forward path."""

from __future__ import annotations

import torch
from torch import nn

from ..common import (
    PLANNING_OUTPUT_KEYS,
    PlanningBatch,
    acceleration_from_velocity,
    build_time_stamps,
    collision_free_mask,
    curvature_from_trajectory,
    kinematic_feasible_mask,
    validate_planning_batch,
    velocity_from_trajectory,
)
from .config import UniADPlanningConfig


class UniADSceneEncoder(nn.Module):
    """Encode ego history, dynamic agents, and map vectors into scene tokens."""

    def __init__(self, cfg: UniADPlanningConfig) -> None:
        super().__init__()
        hidden = cfg.e2e.hidden_dim
        self.ego_proj = nn.Linear(cfg.e2e.state_dim, hidden)
        self.ego_gru = nn.GRU(input_size=hidden, hidden_size=hidden, batch_first=True)
        self.agent_proj = nn.Linear(cfg.e2e.state_dim, hidden)
        self.map_point_proj = nn.Linear(cfg.e2e.map_feat_dim, hidden)
        self.route_proj = nn.Linear(cfg.route_dim, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, batch: PlanningBatch) -> torch.Tensor:
        ego_tokens = self.ego_proj(batch.ego_history)
        _, ego_hidden = self.ego_gru(ego_tokens)
        ego_token = ego_hidden[-1].unsqueeze(1)  # [B, 1, C]

        agent_tokens = self.agent_proj(batch.agent_states)
        map_tokens = self.map_point_proj(batch.map_polylines).mean(dim=2)

        scene_tokens = [ego_token, agent_tokens, map_tokens]
        if batch.route_features is not None:
            scene_tokens.append(self.route_proj(batch.route_features))
        return self.norm(torch.cat(scene_tokens, dim=1))


class UniADPlanningDecoder(nn.Module):
    """Decode planning query tokens from global scene tokens."""

    def __init__(self, cfg: UniADPlanningConfig) -> None:
        super().__init__()
        hidden = cfg.e2e.hidden_dim
        self.query_embedding = nn.Embedding(cfg.num_query_tokens, hidden)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden * 2, hidden),
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.layers = int(cfg.decoder_layers)

    def forward(self, scene_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = scene_tokens.shape[0]
        query = self.query_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
        for _ in range(self.layers):
            attended, _ = self.cross_attn(query=query, key=scene_tokens, value=scene_tokens, need_weights=False)
            query = self.norm1(query + attended)
            query = self.norm2(query + self.ffn(query))
        plan_token = query.mean(dim=1)
        return query, plan_token


class UniADCandidateHead(nn.Module):
    """Generate multimodal planning trajectories and confidence scores."""

    def __init__(self, cfg: UniADPlanningConfig) -> None:
        super().__init__()
        hidden = cfg.e2e.hidden_dim
        k = cfg.e2e.num_candidates
        t = cfg.e2e.future_steps
        self.delta_head = nn.Linear(hidden, k * t * 2)
        self.logit_head = nn.Linear(hidden, k)
        self.max_step = cfg.e2e.max_speed * cfg.e2e.dt
        self.num_candidates = k
        self.future_steps = t

    def forward(self, plan_token: torch.Tensor, start_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = plan_token.shape[0]
        raw = self.delta_head(plan_token).view(batch_size, self.num_candidates, self.future_steps, 2)
        deltas = torch.tanh(raw) * self.max_step
        trajectories = start_xy[:, None, None, :] + torch.cumsum(deltas, dim=2)
        logits = self.logit_head(plan_token)
        return trajectories, deltas, logits


class UniADLite(nn.Module):
    """Educational UniAD-style planning model under `planning/uniad`."""

    def __init__(self, cfg: UniADPlanningConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.scene_encoder = UniADSceneEncoder(cfg)
        self.planning_decoder = UniADPlanningDecoder(cfg)
        self.candidate_head = UniADCandidateHead(cfg)

    def _select_best(
        self,
        *,
        candidate_scores: torch.Tensor,
        feasible_mask: torch.Tensor,
        collision_free: torch.Tensor,
    ) -> torch.Tensor:
        valid = feasible_mask & collision_free
        safe_scores = candidate_scores.masked_fill(~valid, -1.0)
        safe_best = safe_scores.argmax(dim=-1)
        fallback_best = candidate_scores.argmax(dim=-1)
        has_valid = valid.any(dim=-1)
        return torch.where(has_valid, safe_best, fallback_best)

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

        scene_tokens = self.scene_encoder(batch)
        query_tokens, plan_token = self.planning_decoder(scene_tokens)
        start_xy = batch.ego_history[:, -1, :2]
        candidate_trajectories, deltas, candidate_logits = self.candidate_head(plan_token, start_xy)
        candidate_scores = torch.softmax(candidate_logits, dim=-1)

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
        collision_free, min_distance = collision_free_mask(
            candidate_trajectories=candidate_trajectories,
            agent_states=batch.agent_states,
            safety_margin=self.cfg.e2e.safety_margin,
            dt=self.cfg.e2e.dt,
        )
        selected_index = self._select_best(
            candidate_scores=candidate_scores,
            feasible_mask=feasible_mask,
            collision_free=collision_free,
        )
        gather_idx = selected_index[:, None, None, None].expand(-1, 1, self.cfg.e2e.future_steps, 2)
        selected_trajectory = candidate_trajectories.gather(dim=1, index=gather_idx).squeeze(1)
        safety_margin_violation = torch.relu(self.cfg.e2e.safety_margin - min_distance)
        time_stamps = build_time_stamps(
            future_steps=self.cfg.e2e.future_steps,
            dt=self.cfg.e2e.dt,
            device=candidate_trajectories.device,
            dtype=candidate_trajectories.dtype,
        )

        outputs = {
            "scene_tokens": scene_tokens,
            "query_tokens": query_tokens,
            "plan_token": plan_token,
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
            "time_stamps": time_stamps,
        }
        for key in PLANNING_OUTPUT_KEYS:
            if key not in outputs:
                raise KeyError(f"Missing required planning output key: {key}")
        return outputs
