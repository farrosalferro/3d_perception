"""Pure-PyTorch PolarFormer transformer decoder blocks."""

from __future__ import annotations

import torch
from torch import nn


class PolarTransformerDecoderLayerLite(nn.Module):
    """Decoder layer: self-attn -> cross-attn -> FFN."""

    def __init__(self, embed_dims: int, num_heads: int, ffn_dims: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, ffn_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dims, embed_dims),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        *,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = query + query_pos
        query2 = self.self_attn(q, q, query, need_weights=False)[0]
        query = self.norm1(query + self.dropout(query2))

        q = query + query_pos
        k = memory + key_pos
        query2 = self.cross_attn(q, k, memory, key_padding_mask=key_padding_mask, need_weights=False)[0]
        query = self.norm2(query + self.dropout(query2))

        query2 = self.ffn(query)
        query = self.norm3(query + self.dropout(query2))
        return query


class PolarTransformerDecoderLite(nn.Module):
    """Stacked PolarFormer-lite decoder layers."""

    def __init__(
        self,
        num_layers: int,
        *,
        embed_dims: int,
        ffn_dims: int,
        num_heads: int,
        dropout: float = 0.1,
        return_intermediate: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                PolarTransformerDecoderLayerLite(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    ffn_dims=ffn_dims,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.post_norm = nn.LayerNorm(embed_dims)
        self.return_intermediate = return_intermediate

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        *,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.return_intermediate:
            for layer in self.layers:
                query = layer(
                    query,
                    memory,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    key_padding_mask=key_padding_mask,
                )
            return self.post_norm(query).unsqueeze(0)

        intermediate = []
        for layer in self.layers:
            query = layer(
                query,
                memory,
                query_pos=query_pos,
                key_pos=key_pos,
                key_padding_mask=key_padding_mask,
            )
            intermediate.append(self.post_norm(query))
        return torch.stack(intermediate)


class PolarTransformerLite(nn.Module):
    """Flatten multi-level polar features and run decoder."""

    def __init__(self, decoder: PolarTransformerDecoderLite, *, num_feature_levels: int, embed_dims: int) -> None:
        super().__init__()
        self.decoder = decoder
        self.level_embeds = nn.Parameter(torch.randn(num_feature_levels, embed_dims))
        self.reference_points = nn.Linear(embed_dims, 3)

    def forward(
        self,
        mlvl_feats: list[torch.Tensor],
        mlvl_masks: list[torch.Tensor],
        query_embed: torch.Tensor,
        mlvl_pos_embeds: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (len(mlvl_feats) == len(mlvl_masks) == len(mlvl_pos_embeds)):
            raise ValueError("Feature, mask, and positional-encoding levels must match.")

        feat_flatten = []
        mask_flatten = []
        pos_flatten = []
        for lvl, (feat, mask, pos) in enumerate(zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, channels, height, width = feat.shape
            feat_lvl = feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
            mask_lvl = mask.flatten(1)  # [B, H*W]
            pos_lvl = pos.flatten(2).transpose(1, 2) + self.level_embeds[lvl].view(1, 1, -1)
            feat_flatten.append(feat_lvl)
            mask_flatten.append(mask_lvl)
            pos_flatten.append(pos_lvl)

        memory = torch.cat(feat_flatten, dim=1).transpose(0, 1)  # [S, B, C]
        key_padding_mask = torch.cat(mask_flatten, dim=1)  # [B, S]
        key_pos = torch.cat(pos_flatten, dim=1).transpose(0, 1)  # [S, B, C]

        bs = memory.shape[1]
        query_pos, query = torch.chunk(query_embed, 2, dim=1)
        query_pos = query_pos.unsqueeze(1).repeat(1, bs, 1)  # [Q, B, C]
        query = query.unsqueeze(1).repeat(1, bs, 1)  # [Q, B, C]

        init_reference = self.reference_points(query_pos.permute(1, 0, 2)).sigmoid()  # [B, Q, 3]
        outs_dec = self.decoder(
            query,
            memory,
            query_pos=query_pos,
            key_pos=key_pos,
            key_padding_mask=key_padding_mask,
        )
        return outs_dec, init_reference

