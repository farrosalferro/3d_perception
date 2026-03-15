"""End-to-end FlashOcc-style trajectory predictor."""

from __future__ import annotations

import torch
from torch import nn

from .backbone import FlashOccBEVEncoder
from .config import FlashOccConfig
from .head import FlashOccPredictionHead
from .postprocess import FlashOccTrajectoryDecoderLite
from .temporal import FlashOccTemporalMixer


class FlashOccLite(nn.Module):
    """Pure PyTorch occupancy-conditioned trajectory predictor."""

    def __init__(self, cfg: FlashOccConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = FlashOccBEVEncoder(cfg)
        self.temporal_mixer = FlashOccTemporalMixer(cfg)
        self.prediction_head = FlashOccPredictionHead(cfg)
        self.decoder = FlashOccTrajectoryDecoderLite(
            topk=cfg.topk,
            occ_decode_use_gpu=cfg.occupancy_head.decode_use_gpu,
        )

    def forward(
        self,
        occ_seq: torch.Tensor,
        *,
        decode: bool = False,
        history_to_key: torch.Tensor | None = None,
        bda: torch.Tensor | None = None,
        bda_adj: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | dict[str, object]:
        bev_sequence, depth_logits, depth_probs = self.backbone(occ_seq, return_depth=True)
        bev_fused, temporal_tokens, bev_sequence_aligned = self.temporal_mixer(
            bev_sequence,
            history_to_key=history_to_key,
            bda=bda,
            bda_adj=bda_adj,
            return_aligned=True,
        )
        outputs = self.prediction_head(bev_fused)
        outputs.update(
            {
                "bev_sequence": bev_sequence,
                "bev_sequence_aligned": bev_sequence_aligned,
                "bev_fused": bev_fused,
                "temporal_tokens": temporal_tokens,
                "depth_logits": depth_logits,
                "depth_probs": depth_probs,
            }
        )
        if decode:
            decoded = self.decoder.decode(outputs["traj_positions"], outputs["mode_logits"])
            decoded_occ = self.decoder.decode_occupancy(outputs["occupancy_logits"])
            return {"preds": outputs, "decoded": decoded, "decoded_occ": decoded_occ}
        return outputs

    def simple_test(self, occ_seq: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        result = self.forward(occ_seq, decode=True)
        if not isinstance(result, dict):
            raise TypeError("Expected dictionary output from decode path.")
        decoded = result["decoded"]
        if not isinstance(decoded, list):
            raise TypeError("Expected decoded predictions as a list.")
        return decoded

