"""Positional encodings."""

from __future__ import annotations

import torch
from torch import nn


class LearnedPositionalEncoding2D(nn.Module):
    """A lightweight learned 2D positional encoding.

    This mirrors the behavior used by BEVFormer where row and column
    embeddings are concatenated.
    """

    def __init__(self, num_feats: int, row_num_embed: int, col_num_embed: int) -> None:
        super().__init__()
        self.num_feats = num_feats
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """Return positional tensor with shape [B, 2*num_feats, H, W]."""

        if mask.dim() != 3:
            raise ValueError(f"mask must have shape [B, H, W], got {tuple(mask.shape)}")
        bs, h, w = mask.shape
        y = torch.arange(h, device=mask.device)
        x = torch.arange(w, device=mask.device)
        y_embed = self.row_embed(y)  # [H, C]
        x_embed = self.col_embed(x)  # [W, C]

        pos = torch.cat(
            (
                x_embed.unsqueeze(0).repeat(h, 1, 1),
                y_embed.unsqueeze(1).repeat(1, w, 1),
            ),
            dim=-1,
        )  # [H, W, 2C]
        return pos.permute(2, 0, 1).unsqueeze(0).repeat(bs, 1, 1, 1)
