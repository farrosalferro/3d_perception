"""Pure-PyTorch StreamPETR temporal decoder blocks."""

from __future__ import annotations

import torch
from torch import nn


class StreamPETRTemporalDecoderLayerLite(nn.Module):
    """Decoder layer: temporal self-attn -> spatial cross-attn -> FFN."""

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
        temp_memory: torch.Tensor | None = None,
        temp_pos: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = query + query_pos
        temporal_key = query
        temporal_pos = query_pos
        if temp_memory is not None and temp_memory.numel() > 0:
            if temp_pos is None:
                temp_pos = torch.zeros_like(temp_memory)
            temporal_key = torch.cat([query, temp_memory], dim=0)
            temporal_pos = torch.cat([query_pos, temp_pos], dim=0)
        k = temporal_key + temporal_pos
        query2 = self.self_attn(q, k, temporal_key, need_weights=False)[0]
        query = self.norm1(query + self.dropout(query2))

        q = query + query_pos
        k = memory + key_pos
        query2 = self.cross_attn(
            q,
            k,
            memory,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        query = self.norm2(query + self.dropout(query2))

        query2 = self.ffn(query)
        query = self.norm3(query + self.dropout(query2))
        return query


class StreamPETRTemporalDecoderLite(nn.Module):
    """Stacked StreamPETR temporal decoder layers."""

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
                StreamPETRTemporalDecoderLayerLite(
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
        temp_memory: torch.Tensor | None = None,
        temp_pos: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.return_intermediate:
            for layer in self.layers:
                query = layer(
                    query,
                    memory,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    temp_memory=temp_memory,
                    temp_pos=temp_pos,
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
                temp_memory=temp_memory,
                temp_pos=temp_pos,
                key_padding_mask=key_padding_mask,
            )
            intermediate.append(self.post_norm(query))
        return torch.stack(intermediate)


class StreamPETRTemporalTransformerLite(nn.Module):
    """Flatten camera features and run StreamPETR temporal decoder."""

    def __init__(self, decoder: StreamPETRTemporalDecoderLite) -> None:
        super().__init__()
        self.decoder = decoder
        self.embed_dims = decoder.post_norm.normalized_shape[0]

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        query_embed: torch.Tensor,
        pos_embed: torch.Tensor,
        *,
        tgt: torch.Tensor | None = None,
        temp_memory: torch.Tensor | None = None,
        temp_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, Ncam, C, H, W]
            mask: [B, Ncam, H, W], True means invalid.
            query_embed: [B, Q, C]
            pos_embed: [B, Ncam, C, H, W]
            tgt: [B, Q, C] or None
            temp_memory: [B, M, C] or None
            temp_pos: [B, M, C] or None
        Returns:
            outs_dec: [L, Q, B, C]
            memory: [B, Ncam, C, H, W]
        """

        bs, num_cams, channels, height, width = x.shape
        memory = x.permute(1, 3, 4, 0, 2).reshape(num_cams * height * width, bs, channels)
        key_pos = pos_embed.permute(1, 3, 4, 0, 2).reshape(num_cams * height * width, bs, channels)
        key_padding_mask = mask.reshape(bs, num_cams * height * width)

        query_pos = query_embed.transpose(0, 1).contiguous()
        if tgt is None:
            target = torch.zeros_like(query_pos)
        else:
            target = tgt.transpose(0, 1).contiguous()

        temporal_memory = None
        temporal_pos = None
        if temp_memory is not None and temp_memory.numel() > 0:
            temporal_memory = temp_memory.transpose(0, 1).contiguous()
            temporal_pos = torch.zeros_like(temporal_memory) if temp_pos is None else temp_pos.transpose(0, 1).contiguous()

        outs_dec = self.decoder(
            target,
            memory,
            query_pos=query_pos,
            key_pos=key_pos,
            temp_memory=temporal_memory,
            temp_pos=temporal_pos,
            key_padding_mask=key_padding_mask,
        )
        memory = memory.reshape(num_cams, height, width, bs, channels).permute(3, 0, 4, 1, 2)
        return outs_dec, memory

