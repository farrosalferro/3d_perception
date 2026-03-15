"""Pure-PyTorch PolarFormer transformer decoder blocks."""

from __future__ import annotations

import torch
from torch import nn

from .utils import inverse_sigmoid


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
        reference_points: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del reference_points  # kept for API parity with upstream decoder layers
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
        reference_points: torch.Tensor,
        valid_ratios: torch.Tensor,
        reg_branches: nn.ModuleList | None = None,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if reference_points.dim() != 3 or reference_points.shape[-1] != 3:
            raise ValueError(
                f"reference_points must have shape [B, Q, 3], got {tuple(reference_points.shape)}."
            )
        if valid_ratios.dim() != 3 or valid_ratios.shape[-1] != 2:
            raise ValueError(f"valid_ratios must have shape [B, L, 2], got {tuple(valid_ratios.shape)}.")

        if not self.return_intermediate:
            output = query
            for layer_index, layer in enumerate(self.layers):
                reference_points_input = reference_points[:, :, None, :2] * valid_ratios[:, None]
                output = layer(
                    output,
                    memory,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=reference_points_input,
                    key_padding_mask=key_padding_mask,
                )
                output_batch_first = output.permute(1, 0, 2)
                if reg_branches is not None:
                    tmp = reg_branches[layer_index](output_batch_first)
                    new_reference_points = torch.zeros_like(reference_points)
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                    new_reference_points[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
                    reference_points = new_reference_points.sigmoid().detach()
            return self.post_norm(output).unsqueeze(0), reference_points.unsqueeze(0)

        intermediate = []
        intermediate_reference_points = []
        output = query
        for layer_index, layer in enumerate(self.layers):
            reference_points_input = reference_points[:, :, None, :2] * valid_ratios[:, None]
            output = layer(
                output,
                memory,
                query_pos=query_pos,
                key_pos=key_pos,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
            )
            output_batch_first = output.permute(1, 0, 2)
            if reg_branches is not None:
                tmp = reg_branches[layer_index](output_batch_first)
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
                reference_points = new_reference_points.sigmoid().detach()

            intermediate.append(self.post_norm(output))
            intermediate_reference_points.append(reference_points)
        return torch.stack(intermediate), torch.stack(intermediate_reference_points)


class PolarTransformerLite(nn.Module):
    """Flatten multi-level polar features and run decoder."""

    def __init__(self, decoder: PolarTransformerDecoderLite, *, num_feature_levels: int, embed_dims: int) -> None:
        super().__init__()
        self.decoder = decoder
        self.level_embeds = nn.Parameter(torch.randn(num_feature_levels, embed_dims))
        self.reference_points = nn.Linear(embed_dims, 3)

    @staticmethod
    def get_valid_ratio(mask: torch.Tensor) -> torch.Tensor:
        """Get valid width/height ratio from [B, H, W] mask."""

        if mask.dim() != 3:
            raise ValueError(f"mask must have shape [B, H, W], got {tuple(mask.shape)}.")
        _, height, width = mask.shape
        valid_h = torch.sum(~mask[:, :, 0], dim=1)
        valid_w = torch.sum(~mask[:, 0, :], dim=1)
        valid_ratio_h = valid_h.float() / float(height)
        valid_ratio_w = valid_w.float() / float(width)
        return torch.stack((valid_ratio_w, valid_ratio_h), dim=-1)

    def forward(
        self,
        mlvl_feats: list[torch.Tensor],
        mlvl_masks: list[torch.Tensor],
        query_embed: torch.Tensor,
        mlvl_pos_embeds: list[torch.Tensor],
        *,
        reg_branches: nn.ModuleList | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        if not (len(mlvl_feats) == len(mlvl_masks) == len(mlvl_pos_embeds)):
            raise ValueError("Feature, mask, and positional-encoding levels must match.")

        feat_flatten = []
        mask_flatten = []
        pos_flatten = []
        for level_idx, (feat, mask, pos_embed) in enumerate(zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            if feat.dim() != 4:
                raise ValueError(f"mlvl_feats[{level_idx}] must have shape [B, C, H, W], got {tuple(feat.shape)}.")
            if mask.dim() != 3:
                raise ValueError(f"mlvl_masks[{level_idx}] must have shape [B, H, W], got {tuple(mask.shape)}.")
            if pos_embed.dim() != 4:
                raise ValueError(
                    f"mlvl_pos_embeds[{level_idx}] must have shape [B, C, H, W], got {tuple(pos_embed.shape)}."
                )
            if feat.shape[0] != mask.shape[0] or feat.shape[-2:] != mask.shape[-2:]:
                raise ValueError(
                    f"Feature/mask shape mismatch at level {level_idx}: feat={tuple(feat.shape)}, mask={tuple(mask.shape)}."
                )
            if feat.shape != pos_embed.shape:
                raise ValueError(
                    f"Feature/positional shape mismatch at level {level_idx}: "
                    f"feat={tuple(feat.shape)}, pos={tuple(pos_embed.shape)}."
                )
            if feat.shape[1] != self.level_embeds.shape[1]:
                raise ValueError(
                    f"Feature channel mismatch at level {level_idx}: expected {self.level_embeds.shape[1]}, "
                    f"got {feat.shape[1]}."
                )
            feat_lvl = feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
            mask_lvl = mask.flatten(1)  # [B, H*W]
            pos_lvl = pos_embed.flatten(2).transpose(1, 2) + self.level_embeds[level_idx].view(1, 1, -1)
            feat_flatten.append(feat_lvl)
            mask_flatten.append(mask_lvl)
            pos_flatten.append(pos_lvl)
        valid_ratios = torch.stack([self.get_valid_ratio(mask) for mask in mlvl_masks], dim=1)

        memory = torch.cat(feat_flatten, dim=1).permute(1, 0, 2)  # [S, B, C]
        key_padding_mask = torch.cat(mask_flatten, dim=1)  # [B, S]
        key_pos = torch.cat(pos_flatten, dim=1).permute(1, 0, 2)  # [S, B, C]

        batch_size = memory.shape[1]
        query_pos, query = torch.chunk(query_embed, 2, dim=1)
        query_pos = query_pos.unsqueeze(1).repeat(1, batch_size, 1)  # [Q, B, C]
        query = query.unsqueeze(1).repeat(1, batch_size, 1)  # [Q, B, C]

        reference_points = self.reference_points(query_pos.permute(1, 0, 2)).sigmoid()  # [B, Q, 3]
        init_reference = reference_points
        inter_states, inter_references = self.decoder(
            query,
            memory,
            reference_points=reference_points,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            query_pos=query_pos,
            key_pos=key_pos,
            key_padding_mask=key_padding_mask,
        )
        return inter_states, init_reference, inter_references, None, None
