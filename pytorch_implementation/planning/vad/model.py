"""Pure-PyTorch VAD-style end-to-end planning forward path."""

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
from .config import VADPlanningConfig


class VADVectorEncoder(nn.Module):
    """Encode vectorized scene elements (ego, agents, lane polylines)."""

    def __init__(self, cfg: VADPlanningConfig) -> None:
        super().__init__()
        hidden = cfg.e2e.hidden_dim
        self.ego_proj = nn.Linear(cfg.e2e.state_dim, hidden)
        self.agent_proj = nn.Linear(cfg.e2e.state_dim, hidden)
        self.map_point_proj = nn.Linear(cfg.e2e.map_feat_dim, hidden)
        self.route_proj = nn.Linear(cfg.lane_token_dim, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, batch: PlanningBatch) -> tuple[torch.Tensor, torch.Tensor]:
        ego_token = self.ego_proj(batch.ego_history[:, -1]).unsqueeze(1)
        agent_tokens = self.agent_proj(batch.agent_states)
        map_tokens = self.map_point_proj(batch.map_polylines).mean(dim=2)
        vector_tokens = [ego_token, agent_tokens, map_tokens]
        if batch.route_features is not None:
            vector_tokens.append(self.route_proj(batch.route_features))
        tokens = self.norm(torch.cat(vector_tokens, dim=1))
        lane_points = batch.map_polylines[..., :2]
        return tokens, lane_points


class VADPlannerCore(nn.Module):
    """Vectorized interaction planner that outputs multimodal trajectories."""

    def __init__(self, cfg: VADPlanningConfig) -> None:
        super().__init__()
        self.cfg = cfg
        hidden = cfg.e2e.hidden_dim
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden, hidden * 2),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(hidden * 2, hidden),
                )
                for _ in range(cfg.interaction_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden)
        self.traj_head = nn.Linear(hidden, cfg.e2e.num_candidates * cfg.e2e.future_steps * 2)
        self.score_head = nn.Linear(hidden, cfg.e2e.num_candidates)
        self.max_step = cfg.e2e.max_speed * cfg.e2e.dt

    def forward(self, vector_tokens: torch.Tensor, start_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ego_query = vector_tokens[:, :1, :]
        fused = ego_query
        for layer in self.layers:
            attended, _ = self.cross_attn(query=fused, key=vector_tokens, value=vector_tokens, need_weights=False)
            fused = self.norm(fused + attended + layer(fused))
        fused = fused.squeeze(1)

        batch_size = fused.shape[0]
        raw_deltas = self.traj_head(fused).view(
            batch_size,
            self.cfg.e2e.num_candidates,
            self.cfg.e2e.future_steps,
            2,
        )
        deltas = torch.tanh(raw_deltas) * self.max_step
        trajectories = start_xy[:, None, None, :] + torch.cumsum(deltas, dim=2)
        logits = self.score_head(fused)
        return trajectories, deltas, logits


class VADLite(nn.Module):
    """Educational VAD-style planning model under `planning/vad`."""

    def __init__(self, cfg: VADPlanningConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vector_encoder = VADVectorEncoder(cfg)
        self.planner_core = VADPlannerCore(cfg)

    @staticmethod
    def _lane_direction(lane_points: torch.Tensor) -> torch.Tensor:
        # lane_points: [B, P, S, 2]
        direction = lane_points[:, :, -1, :] - lane_points[:, :, 0, :]
        direction = direction.mean(dim=1)
        return direction / torch.linalg.norm(direction, dim=-1, keepdim=True).clamp(min=1e-6)

    @staticmethod
    def _trajectory_direction(trajectories: torch.Tensor) -> torch.Tensor:
        direction = trajectories[:, :, -1, :] - trajectories[:, :, 0, :]
        return direction / torch.linalg.norm(direction, dim=-1, keepdim=True).clamp(min=1e-6)

    @staticmethod
    def _lane_distance(trajectories: torch.Tensor, lane_points: torch.Tensor) -> torch.Tensor:
        # [B, K, T, 1, 1, 2] - [B, 1, 1, P, S, 2] -> [B, K, T, P, S]
        diff = trajectories[:, :, :, None, None, :] - lane_points[:, None, None, :, :, :]
        distances = torch.linalg.norm(diff, dim=-1)
        return distances.amin(dim=(-1, -2, -3))

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

        vector_tokens, lane_points = self.vector_encoder(batch)
        start_xy = batch.ego_history[:, -1, :2]
        candidate_trajectories, deltas, candidate_logits = self.planner_core(vector_tokens, start_xy)
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

        lane_distance = self._lane_distance(candidate_trajectories, lane_points)
        lane_tolerance = self.cfg.e2e.safety_margin * 3.0
        lane_mask = lane_distance <= lane_tolerance

        lane_direction = self._lane_direction(lane_points)
        trajectory_direction = self._trajectory_direction(candidate_trajectories)
        lane_alignment = (trajectory_direction * lane_direction[:, None, :]).sum(dim=-1)
        lane_align_mask = lane_alignment >= 0.0

        valid = feasible_mask & collision_free & lane_mask & lane_align_mask
        safe_scores = candidate_scores.masked_fill(~valid, -1.0)
        safe_best = safe_scores.argmax(dim=-1)
        fallback_best = candidate_scores.argmax(dim=-1)
        selected_index = torch.where(valid.any(dim=-1), safe_best, fallback_best)
        gather_idx = selected_index[:, None, None, None].expand(-1, 1, self.cfg.e2e.future_steps, 2)
        selected_trajectory = candidate_trajectories.gather(dim=1, index=gather_idx).squeeze(1)

        safety_margin_violation = torch.relu(self.cfg.e2e.safety_margin - min_distance)
        constraint_cost = (
            self.cfg.collision_weight * safety_margin_violation
            + self.cfg.boundary_weight * lane_distance
            + self.cfg.lane_align_weight * torch.relu(1.0 - lane_alignment)
        )
        time_stamps = build_time_stamps(
            future_steps=self.cfg.e2e.future_steps,
            dt=self.cfg.e2e.dt,
            device=candidate_trajectories.device,
            dtype=candidate_trajectories.dtype,
        )

        outputs = {
            "vector_tokens": vector_tokens,
            "lane_points": lane_points,
            "candidate_logits": candidate_logits,
            "candidate_deltas": deltas,
            "candidate_trajectories": candidate_trajectories,
            "candidate_scores": candidate_scores,
            "selected_index": selected_index,
            "selected_trajectory": selected_trajectory,
            "velocity": velocity,
            "acceleration": acceleration,
            "curvature": curvature,
            "feasible_mask": feasible_mask & lane_mask & lane_align_mask,
            "collision_free_mask": collision_free,
            "safety_margin_violation": safety_margin_violation,
            "min_distance": min_distance,
            "lane_distance": lane_distance,
            "lane_alignment": lane_alignment,
            "constraint_cost": constraint_cost,
            "time_stamps": time_stamps,
        }
        for key in PLANNING_OUTPUT_KEYS:
            if key not in outputs:
                raise KeyError(f"Missing required planning output key: {key}")
        return outputs
