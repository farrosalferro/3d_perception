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
        self.decoder = FlashOccTrajectoryDecoderLite(topk=cfg.topk)

    def forward(
        self,
        occ_seq: torch.Tensor,
        *,
        decode: bool = False,
    ) -> dict[str, torch.Tensor] | dict[str, object]:
        bev_sequence = self.backbone(occ_seq)
        bev_fused, temporal_tokens = self.temporal_mixer(bev_sequence)
        outputs = self.prediction_head(bev_fused)
        outputs.update(
            {
                "bev_sequence": bev_sequence,
                "bev_fused": bev_fused,
                "temporal_tokens": temporal_tokens,
            }
        )
        if decode:
            decoded = self.decoder.decode(outputs["traj_positions"], outputs["mode_logits"])
            return {"preds": outputs, "decoded": decoded}
        return outputs

    def simple_test(self, occ_seq: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        result = self.forward(occ_seq, decode=True)
        if not isinstance(result, dict):
            raise TypeError("Expected dictionary output from decode path.")
        decoded = result["decoded"]
        if not isinstance(decoded, list):
            raise TypeError("Expected decoded predictions as a list.")
        return decoded

