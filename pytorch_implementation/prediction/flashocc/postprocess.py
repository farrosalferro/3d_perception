"""Postprocess helpers for FlashOcc-style trajectory predictions."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FlashOccTrajectoryDecoderLite:
    """Decode multi-modal trajectories into top-k query predictions."""

    topk: int
    occ_decode_use_gpu: bool = True

    def decode_single(self, traj_positions: torch.Tensor, mode_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        # traj_positions: [Q, M, H, 2], mode_logits: [Q, M]
        mode_probs = mode_logits.softmax(dim=-1)
        best_mode_scores, best_mode_indices = mode_probs.max(dim=-1)
        query_count = int(traj_positions.shape[0])
        topk = min(self.topk, query_count)
        top_scores, top_query_indices = best_mode_scores.topk(topk, dim=0)
        top_mode_indices = best_mode_indices[top_query_indices]

        gather_traj = traj_positions[top_query_indices]
        best_traj = gather_traj[torch.arange(topk, device=traj_positions.device), top_mode_indices]
        return {
            "trajectories": best_traj,
            "scores": top_scores,
            "mode_indices": top_mode_indices,
            "query_indices": top_query_indices,
        }

    def decode(self, traj_positions: torch.Tensor, mode_logits: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        batch_size = int(traj_positions.shape[0])
        return [self.decode_single(traj_positions[idx], mode_logits[idx]) for idx in range(batch_size)]

    def get_occ(self, occupancy_logits: torch.Tensor) -> list[torch.Tensor]:
        """CPU decode contract: list of uint8 occupancy labels [Dx, Dy, Dz]."""

        occ_score = occupancy_logits.softmax(dim=-1)
        occ_res = occ_score.argmax(dim=-1).to(dtype=torch.uint8).cpu()
        return [occ_res[idx] for idx in range(int(occ_res.shape[0]))]

    def get_occ_gpu(self, occupancy_logits: torch.Tensor) -> list[torch.Tensor]:
        """GPU decode contract: list of int occupancy labels [Dx, Dy, Dz]."""

        occ_score = occupancy_logits.softmax(dim=-1)
        occ_res = occ_score.argmax(dim=-1).int()
        return [occ_res[idx] for idx in range(int(occ_res.shape[0]))]

    def decode_occupancy(self, occupancy_logits: torch.Tensor) -> list[torch.Tensor]:
        if self.occ_decode_use_gpu:
            return self.get_occ_gpu(occupancy_logits)
        return self.get_occ(occupancy_logits)

