"""Task head contracts for pure-PyTorch BEVerse-lite."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ...common.postprocess.gather import gather_mode_trajectories
from .config import BEVerseForwardConfig


class TrajectoryHeadLite(nn.Module):
    """Predict multimodal trajectories and expose motion decode contracts."""

    def __init__(self, cfg: BEVerseForwardConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.shared = nn.Sequential(
            nn.LayerNorm(cfg.embed_dims),
            nn.Linear(cfg.embed_dims, cfg.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.embed_dims, cfg.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.delta_head = nn.Linear(cfg.embed_dims, cfg.num_modes * 2)
        self.mode_head = nn.Linear(cfg.embed_dims, cfg.num_modes)

    def forward(
        self,
        temporal_tokens: torch.Tensor,
        targets: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        del targets
        if temporal_tokens.dim() != 3:
            raise ValueError(
                f"temporal_tokens must be [B, T, C], got {tuple(temporal_tokens.shape)}"
            )
        batch_size, horizon, channels = temporal_tokens.shape
        if channels != self.cfg.embed_dims:
            raise ValueError(
                f"Expected temporal_tokens channels {self.cfg.embed_dims}, got {channels}"
            )
        if horizon != self.cfg.pred_horizon:
            raise ValueError(
                f"Expected horizon {self.cfg.pred_horizon}, got {horizon}"
            )

        shared_tokens = self.shared(temporal_tokens)
        raw_deltas = self.delta_head(shared_tokens)
        raw_deltas = raw_deltas.view(batch_size, horizon, self.cfg.num_modes, 2)
        trajectory_deltas = torch.tanh(raw_deltas).permute(0, 2, 1, 3) * self.cfg.max_delta
        trajectory_preds = trajectory_deltas.cumsum(dim=2)

        mode_logits = self.mode_head(shared_tokens[:, -1, :])
        mode_probs = torch.softmax(mode_logits, dim=-1)
        return {
            "trajectory_deltas": trajectory_deltas,
            "trajectory_preds": trajectory_preds,
            "mode_logits": mode_logits,
            "mode_probs": mode_probs,
            "mode_log_probs": torch.log_softmax(mode_logits, dim=-1),
        }

    def loss(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if targets is None or "target_trajectory" not in targets:
            return {}

        pred_modes = predictions["trajectory_preds"]
        mode_logits = predictions["mode_logits"]
        target = targets["target_trajectory"]
        if pred_modes.dim() != 4 or pred_modes.shape[-1] != 2:
            raise ValueError("trajectory_preds must have shape [B, M, T, 2].")
        batch_size, _, horizon, _ = pred_modes.shape
        if target.shape != (batch_size, horizon, 2):
            raise ValueError(
                f"target_trajectory must be {(batch_size, horizon, 2)}, got {tuple(target.shape)}"
            )

        valid_mask = targets.get("target_valid_mask")
        if valid_mask is None:
            mask_bt = torch.ones(
                (batch_size, horizon), device=pred_modes.device, dtype=pred_modes.dtype
            )
        else:
            if valid_mask.shape != (batch_size, horizon):
                raise ValueError(
                    f"target_valid_mask must be {(batch_size, horizon)}, got {tuple(valid_mask.shape)}"
                )
            mask_bt = valid_mask.to(device=pred_modes.device, dtype=pred_modes.dtype)

        target_expanded = target.unsqueeze(1)
        l2_per_step = torch.linalg.norm(pred_modes - target_expanded, dim=-1)
        mask_bmt = mask_bt.unsqueeze(1)
        denom = mask_bmt.sum(dim=-1).clamp_min(1.0)
        ade_per_mode = (l2_per_step * mask_bmt).sum(dim=-1) / denom
        best_mode_idx = ade_per_mode.argmin(dim=-1)

        best_traj = gather_mode_trajectories(pred_modes, best_mode_idx, mode_dim=1).squeeze(1)

        l1_per_step = (best_traj - target).abs().sum(dim=-1)
        reg_loss = (l1_per_step * mask_bt).sum() / mask_bt.sum().clamp_min(1.0)
        cls_loss = F.cross_entropy(mode_logits, best_mode_idx)
        return {
            "loss_motion_reg": reg_loss,
            "loss_motion_cls": cls_loss,
        }

    def inference(self, predictions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mode_probs = predictions["mode_probs"]
        trajectory_preds = predictions["trajectory_preds"]
        if mode_probs.dim() != 2:
            raise ValueError(f"mode_probs must be [B, M], got {tuple(mode_probs.shape)}")
        if trajectory_preds.dim() != 4:
            raise ValueError(
                f"trajectory_preds must be [B, M, T, 2], got {tuple(trajectory_preds.shape)}"
            )

        batch_size, num_modes = mode_probs.shape
        _, _, horizon, coord_dim = trajectory_preds.shape
        if coord_dim != 2:
            raise ValueError("trajectory_preds must have XY coordinate dimension 2.")

        topk = min(self.cfg.decode_topk, num_modes)
        topk_mode_prob, topk_mode_idx = torch.topk(mode_probs, k=topk, dim=-1)
        topk_trajectory = gather_mode_trajectories(trajectory_preds, topk_mode_idx, mode_dim=1)

        best_mode_idx = topk_mode_idx[:, 0]
        best_mode_prob = topk_mode_prob[:, 0]
        best_trajectory = topk_trajectory[:, 0]
        return {
            "best_mode_idx": best_mode_idx,
            "best_mode_prob": best_mode_prob,
            "best_trajectory": best_trajectory,
            "topk_mode_idx": topk_mode_idx,
            "topk_mode_prob": topk_mode_prob,
            "topk_trajectory": topk_trajectory,
        }


class _MapHeadStub(nn.Module):
    """Minimal map decoder placeholder for BEVerse task contracts."""

    def __init__(self, cfg: BEVerseForwardConfig, num_map_classes: int = 3) -> None:
        super().__init__()
        self.num_map_classes = num_map_classes
        self.logits = nn.Conv2d(cfg.embed_dims, num_map_classes, kernel_size=1)

    def forward(self, bev_feat: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"semantic_logits": self.logits(bev_feat)}

    def loss(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if targets is None or "semantic_map" not in targets:
            return {}
        semantic_map = targets["semantic_map"]
        logits = predictions["semantic_logits"]
        if semantic_map.dim() == 4 and semantic_map.shape[1] == 1:
            semantic_map = semantic_map[:, 0]
        if semantic_map.dim() != 3:
            return {}
        if semantic_map.shape[0] != logits.shape[0]:
            return {}
        loss = F.cross_entropy(logits, semantic_map.long())
        return {"loss_map_seg": loss}

    def inference(self, predictions: dict[str, torch.Tensor]) -> torch.Tensor:
        return predictions["semantic_logits"].argmax(dim=1, keepdim=True)


class _DetectionHeadStub(nn.Module):
    """Minimal 3D detection placeholder matching key decode contracts."""

    def __init__(self, cfg: BEVerseForwardConfig) -> None:
        super().__init__()
        self.heatmap = nn.Conv2d(cfg.embed_dims, 1, kernel_size=1)
        self.reg = nn.Conv2d(cfg.embed_dims, 2, kernel_size=1)

    def forward(self, bev_feat: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        return [{"heatmap": self.heatmap(bev_feat), "reg": self.reg(bev_feat)}]

    def loss(
        self,
        predictions: list[dict[str, torch.Tensor]],
        targets: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        del targets
        heatmap = predictions[0]["heatmap"]
        return {"loss_3dod_heatmap_reg": heatmap.new_zeros(())}

    def get_bboxes(
        self,
        predictions: list[dict[str, torch.Tensor]],
        img_metas: list[dict[str, Any]] | None,
        rescale: bool = False,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        del predictions, img_metas, rescale
        return []


class MultiTaskHeadLite(nn.Module):
    """Pure-PyTorch multi-task head with BEVerse-like contracts."""

    def __init__(self, cfg: BEVerseForwardConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.task_names_ordered = ["map", "3dod", "motion"]
        self.task_enable = {
            "map": cfg.task_enable_map,
            "3dod": cfg.task_enable_3dod,
            "motion": cfg.task_enable_motion,
        }
        self.task_weights = {
            "map": cfg.task_weight_map,
            "3dod": cfg.task_weight_3dod,
            "motion": cfg.task_weight_motion,
        }

        self.task_decoders = nn.ModuleDict()
        if self.task_enable["map"]:
            self.task_decoders["map"] = _MapHeadStub(cfg)
        if self.task_enable["3dod"]:
            self.task_decoders["3dod"] = _DetectionHeadStub(cfg)
        if self.task_enable["motion"]:
            self.task_decoders["motion"] = TrajectoryHeadLite(cfg)

    def scale_task_losses(
        self,
        task_name: str,
        task_loss_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not task_loss_dict:
            return {}
        weight = self.task_weights.get(task_name, 1.0)
        scaled = {key: value * weight for key, value in task_loss_dict.items()}
        scaled[f"{task_name}_sum"] = sum(scaled.values())
        return scaled

    def loss(
        self,
        predictions: dict[str, Any],
        targets: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        loss_dict: dict[str, torch.Tensor] = {}
        if self.task_enable["3dod"] and "3dod" in predictions:
            det_loss = self.task_decoders["3dod"].loss(predictions["3dod"], targets)
            loss_dict.update(self.scale_task_losses("3dod", det_loss))
        if self.task_enable["map"] and "map" in predictions:
            map_loss = self.task_decoders["map"].loss(predictions["map"], targets)
            loss_dict.update(self.scale_task_losses("map", map_loss))
        if self.task_enable["motion"] and "motion" in predictions:
            motion_loss = self.task_decoders["motion"].loss(predictions["motion"], targets)
            loss_dict.update(self.scale_task_losses("motion", motion_loss))
        return loss_dict

    def inference(
        self,
        predictions: dict[str, Any],
        img_metas: list[dict[str, Any]] | None,
        rescale: bool = False,
    ) -> dict[str, Any]:
        del rescale
        outputs: dict[str, Any] = {}
        if self.task_enable["3dod"] and "3dod" in predictions:
            outputs["bbox_list"] = self.task_decoders["3dod"].get_bboxes(
                predictions["3dod"], img_metas=img_metas
            )
        if self.task_enable["map"] and "map" in predictions:
            outputs["pred_semantic_indices"] = self.task_decoders["map"].inference(predictions["map"])
        if self.task_enable["motion"] and "motion" in predictions:
            motion_pred = predictions["motion"]
            motion_decoded = self.task_decoders["motion"].inference(motion_pred)
            outputs["motion_predictions"] = motion_pred
            outputs["motion_decoded"] = motion_decoded
            outputs["motion_segmentation"] = motion_decoded["best_mode_idx"].unsqueeze(-1)
            outputs["motion_instance"] = motion_decoded["best_trajectory"]
        return outputs

    def forward(
        self,
        bev_feats: torch.Tensor,
        targets: dict[str, torch.Tensor] | None = None,
        *,
        temporal_tokens: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        predictions: dict[str, Any] = {}
        if self.task_enable["map"]:
            predictions["map"] = self.task_decoders["map"](bev_feats)
        if self.task_enable["3dod"]:
            predictions["3dod"] = self.task_decoders["3dod"](bev_feats)
        if self.task_enable["motion"]:
            if temporal_tokens is None:
                raise ValueError("temporal_tokens is required when motion task is enabled.")
            predictions["motion"] = self.task_decoders["motion"](temporal_tokens, targets=targets)
        return predictions
