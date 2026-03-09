"""Sparse4D-style learnable instance feature and anchor bank."""

from __future__ import annotations

import torch
from torch import nn


class InstanceBankLite(nn.Module):
    """Holds learnable instance features and 3D anchors."""

    def __init__(self, num_queries: int, embed_dims: int, box_code_size: int) -> None:
        super().__init__()
        self.num_queries = int(num_queries)
        self.embed_dims = int(embed_dims)
        self.box_code_size = int(box_code_size)
        self.instance_feature = nn.Embedding(self.num_queries, self.embed_dims)
        self.anchor = nn.Embedding(self.num_queries, self.box_code_size)
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.instance_feature.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.anchor.weight)
        with torch.no_grad():
            if self.box_code_size >= 6:
                self.anchor.weight[:, 3:6] = 1.0
            if self.box_code_size >= 8:
                self.anchor.weight[:, 7] = 1.0

    def forward(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.instance_feature.weight.unsqueeze(0).expand(batch_size, -1, -1)
        anchors = self.anchor.weight.unsqueeze(0).expand(batch_size, -1, -1)
        if device is not None:
            features = features.to(device=device)
            anchors = anchors.to(device=device)
        if dtype is not None:
            features = features.to(dtype=dtype)
            anchors = anchors.to(dtype=dtype)
        return features, anchors
