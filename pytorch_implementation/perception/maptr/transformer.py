"""Pure-PyTorch MapTR perception transformer blocks."""

from __future__ import annotations

import torch
from torch import nn

from .utils import inverse_sigmoid


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
        reference_points: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Kept for parity with upstream decoder API; this lightweight layer does
        # not consume explicit per-level reference points in attention.
        del reference_points

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
    """Stacked MapTR decoder layers with iterative reference-point updates."""

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

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        *,
        query_pos: torch.Tensor,
        reference_points: torch.Tensor,
        key_pos: torch.Tensor,
        reg_branches: nn.ModuleList | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if reference_points.dim() != 3 or reference_points.shape[-1] != 2:
            raise ValueError(
                f"reference_points must have shape [B, Q, 2], got {tuple(reference_points.shape)}"
            )

        if not self.return_intermediate:
            output = query
            for lid, layer in enumerate(self.layers):
                output = layer(
                    output,
                    memory,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    key_padding_mask=key_padding_mask,
                    reference_points=reference_points[..., :2].unsqueeze(2),
                )
                if reg_branches is not None:
                    tmp = reg_branches[lid](output.permute(1, 0, 2))
                    new_reference_points = torch.zeros_like(reference_points)
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                    reference_points = new_reference_points.sigmoid().detach()
            return self.post_norm(output).unsqueeze(0), reference_points.unsqueeze(0)

        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            output = layer(
                output,
                memory,
                query_pos=query_pos,
                key_pos=key_pos,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points[..., :2].unsqueeze(2),
            )
            output_bs = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output_bs)
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                reference_points = new_reference_points.sigmoid().detach()

            output = output_bs.permute(1, 0, 2)
            intermediate.append(self.post_norm(output))
            intermediate_reference_points.append(reference_points)

        return torch.stack(intermediate), torch.stack(intermediate_reference_points)


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

    def __init__(self, decoder: MapTRDecoderLite, bev_encoder: MapTRBEVEncoderLite, num_cams: int) -> None:
        super().__init__()
        self.decoder = decoder
        self.bev_encoder = bev_encoder
        self.embed_dims = decoder.post_norm.normalized_shape[0]
        self.num_cams = int(num_cams)
        self.level_embeds = nn.Parameter(torch.zeros(1, self.embed_dims))
        self.cams_embeds = nn.Parameter(torch.zeros(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.level_embeds)
        nn.init.normal_(self.cams_embeds)
        nn.init.xavier_uniform_(self.reference_points.weight)
        nn.init.constant_(self.reference_points.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        object_query_embed: torch.Tensor,
        cam_pos_embed: torch.Tensor,
        bev_queries: torch.Tensor,
        bev_pos: torch.Tensor,
        *,
        reg_branches: nn.ModuleList | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, Ncam, C, H, W]
            mask: [B, Ncam, H, W], True means invalid
            object_query_embed: [Q, 2C]
            cam_pos_embed: [B, Ncam, C, H, W]
            bev_queries: [BHW, C]
            bev_pos: [B, BHW, C]
        Returns:
            bev_embed: [B, BHW, C]
            inter_states: [L, Q, B, C]
            init_reference: [B, Q, 2]
            inter_references: [L, B, Q, 2]
        """
        if x.dim() != 5:
            raise ValueError(f"x must have shape [B, Ncam, C, H, W], got {tuple(x.shape)}")
        if mask.dim() != 4:
            raise ValueError(f"mask must have shape [B, Ncam, H, W], got {tuple(mask.shape)}")

        bs, num_cams, channels, height, width = x.shape
        if channels != self.embed_dims:
            raise ValueError(f"Expected channel size {self.embed_dims}, got {channels}")
        if num_cams > self.num_cams:
            raise ValueError(
                f"Input has {num_cams} cameras but transformer is configured for {self.num_cams}"
            )
        if cam_pos_embed.shape != x.shape:
            raise ValueError(
                f"cam_pos_embed must match x shape {tuple(x.shape)}, got {tuple(cam_pos_embed.shape)}"
            )
        if mask.shape != (bs, num_cams, height, width):
            raise ValueError(
                f"mask must have shape {(bs, num_cams, height, width)}, got {tuple(mask.shape)}"
            )
        if object_query_embed.dim() != 2 or object_query_embed.shape[-1] != 2 * self.embed_dims:
            raise ValueError(
                "object_query_embed must have shape [Q, 2*embed_dims], "
                f"got {tuple(object_query_embed.shape)}"
            )

        cam_embed = self.cams_embeds[:num_cams].to(dtype=x.dtype).view(1, num_cams, channels, 1, 1)
        lvl_embed = self.level_embeds[0].to(dtype=x.dtype).view(1, 1, channels, 1, 1)
        x = x + cam_embed + lvl_embed

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

        query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=-1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos).sigmoid()
        init_reference = reference_points

        inter_states, inter_references = self.decoder(
            query=query.permute(1, 0, 2),
            memory=bev_tokens,
            query_pos=query_pos.permute(1, 0, 2),
            reference_points=reference_points,
            key_pos=bev_pos.permute(1, 0, 2),
            reg_branches=reg_branches,
            key_padding_mask=None,
        )
        bev_embed = bev_tokens.permute(1, 0, 2).contiguous()
        return bev_embed, inter_states, init_reference, inter_references
