"""Pure-PyTorch PETR transformer decoder blocks."""

from __future__ import annotations

import torch
from torch import nn


class PETRTransformerDecoderLayerLite(nn.Module):
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
        **kwargs: object,
    ) -> torch.Tensor:
        del kwargs
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


class PETRTransformerDecoderLite(nn.Module):
    """Stacked PETR decoder layers."""

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
                PETRTransformerDecoderLayerLite(
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
        reg_branch: nn.ModuleList | None = None,
    ) -> torch.Tensor:
        del reg_branch
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


class PETRTransformerLite(nn.Module):
    """Flatten camera features and run PETR decoder."""

    def __init__(self, decoder: PETRTransformerDecoderLite) -> None:
        super().__init__()
        self.decoder = decoder
        self.embed_dims = decoder.post_norm.normalized_shape[0]

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        query_embed: torch.Tensor,
        pos_embed: torch.Tensor,
        reg_branch: nn.ModuleList | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, Ncam, C, H, W]
            mask: [B, Ncam, H, W], True means invalid.
            query_embed: [Q, C]
            pos_embed: [B, Ncam, C, H, W]
        Returns:
            outs_dec: [L, B, Q, C]
            memory: [B, Ncam, C, H, W]
        """

        bs, num_cams, channels, height, width = x.shape
        memory = x.permute(1, 3, 4, 0, 2).reshape(num_cams * height * width, bs, channels)
        key_pos = pos_embed.permute(1, 3, 4, 0, 2).reshape(num_cams * height * width, bs, channels)
        key_padding_mask = mask.reshape(bs, num_cams * height * width)

        query_pos = query_embed.unsqueeze(1).repeat(1, bs, 1)
        target = torch.zeros_like(query_pos)
        outs_dec = self.decoder(
            target,
            memory,
            query_pos=query_pos,
            key_pos=key_pos,
            key_padding_mask=key_padding_mask,
            reg_branch=reg_branch,
        )
        outs_dec = outs_dec.transpose(1, 2)
        memory = memory.reshape(num_cams, height, width, bs, channels).permute(3, 0, 4, 1, 2)
        return outs_dec, memory

