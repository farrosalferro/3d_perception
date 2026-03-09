"""Pure-PyTorch MapTR perception transformer blocks."""

from __future__ import annotations

import torch
from torch import nn


class MapTRDecoderLayerLite(nn.Module):
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


class MapTRDecoderLite(nn.Module):
    """Stacked MapTR decoder layers."""

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
                MapTRDecoderLayerLite(
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


class MapTRBEVEncoderLite(nn.Module):
    """Single-layer BEV encoder that attends BEV tokens to camera memory."""

    def __init__(self, embed_dims: int, num_heads: int, ffn_dims: int, dropout: float) -> None:
        super().__init__()
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

    def forward(
        self,
        bev_tokens: torch.Tensor,
        cam_memory: torch.Tensor,
        cam_pos: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out = self.cross_attn(
            bev_tokens,
            cam_memory + cam_pos,
            cam_memory,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        bev_tokens = self.norm1(bev_tokens + self.dropout(attn_out))
        ffn_out = self.ffn(bev_tokens)
        bev_tokens = self.norm2(bev_tokens + self.dropout(ffn_out))
        return bev_tokens


class MapTRPerceptionTransformerLite(nn.Module):
    """Build BEV tokens from multi-view memory and decode map queries."""

    def __init__(self, decoder: MapTRDecoderLite, bev_encoder: MapTRBEVEncoderLite) -> None:
        super().__init__()
        self.decoder = decoder
        self.bev_encoder = bev_encoder
        self.embed_dims = decoder.post_norm.normalized_shape[0]

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        query_embed: torch.Tensor,
        cam_pos_embed: torch.Tensor,
        bev_queries: torch.Tensor,
        bev_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, Ncam, C, H, W]
            mask: [B, Ncam, H, W], True means invalid
            query_embed: [Q, C]
            cam_pos_embed: [B, Ncam, C, H, W]
            bev_queries: [BHW, C]
            bev_pos: [B, BHW, C]
        Returns:
            bev_embed: [B, BHW, C]
            outs_dec: [L, Q, B, C]
        """

        bs, num_cams, channels, height, width = x.shape
        cam_memory = x.permute(1, 3, 4, 0, 2).reshape(num_cams * height * width, bs, channels)
        cam_pos = cam_pos_embed.permute(1, 3, 4, 0, 2).reshape(num_cams * height * width, bs, channels)
        key_padding_mask = mask.reshape(bs, num_cams * height * width)

        bev_tokens = bev_queries.unsqueeze(1).repeat(1, bs, 1) + bev_pos.permute(1, 0, 2)
        bev_tokens = self.bev_encoder(
            bev_tokens,
            cam_memory,
            cam_pos,
            key_padding_mask=key_padding_mask,
        )

        query_pos = query_embed.unsqueeze(1).repeat(1, bs, 1)
        target = torch.zeros_like(query_pos)
        outs_dec = self.decoder(
            target,
            bev_tokens,
            query_pos=query_pos,
            key_pos=bev_pos.permute(1, 0, 2),
            key_padding_mask=None,
        )
        bev_embed = bev_tokens.permute(1, 0, 2).contiguous()
        return bev_embed, outs_dec
