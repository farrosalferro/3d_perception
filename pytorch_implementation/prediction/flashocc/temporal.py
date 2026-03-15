"""Temporal fusion for BEV occupancy features."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import FlashOccConfig


class FlashOccTemporalMixer(nn.Module):
    """Fuse history along time axis with BEVDet4D-style optional warp alignment."""

    def __init__(self, cfg: FlashOccConfig) -> None:
        super().__init__()
        self.cfg = cfg
        embed_dims = cfg.backbone.embed_dims
        self.temporal_conv = nn.Conv1d(embed_dims, embed_dims, kernel_size=3, padding=1, bias=False)
        self.norm = nn.BatchNorm1d(embed_dims)
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dims),
            nn.GELU(),
        )
        self._grid_cache: tuple[int, int, torch.dtype, torch.device, torch.Tensor] | None = None

    def _base_grid(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._grid_cache is not None:
            ch, cw, cdtype, cdevice, cached = self._grid_cache
            if ch == height and cw == width and cdtype == dtype and cdevice == device:
                return cached

        xs = torch.linspace(0, width - 1, width, dtype=dtype, device=device).view(1, width).expand(height, width)
        ys = torch.linspace(0, height - 1, height, dtype=dtype, device=device).view(height, 1).expand(height, width)
        grid = torch.stack((xs, ys, torch.ones_like(xs)), dim=-1)  # [H, W, 3]
        self._grid_cache = (height, width, dtype, device, grid)
        return grid

    def _feature_to_bev_transform(
        self,
        width: int,
        height: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        x_min, x_max = self.cfg.temporal_warp.x_bounds
        y_min, y_max = self.cfg.temporal_warp.y_bounds
        dx = (float(x_max) - float(x_min)) / max(float(width), 1.0)
        dy = (float(y_max) - float(y_min)) / max(float(height), 1.0)
        feat2bev = torch.zeros((3, 3), dtype=dtype, device=device)
        feat2bev[0, 0] = dx
        feat2bev[1, 1] = dy
        feat2bev[0, 2] = float(x_min)
        feat2bev[1, 2] = float(y_min)
        feat2bev[2, 2] = 1.0
        return feat2bev

    def _planar_transform(self, mat: torch.Tensor) -> torch.Tensor:
        if mat.shape[-2:] == (4, 4):
            return mat[..., [0, 1, 3], :][..., [0, 1, 3]]
        if mat.shape[-2:] == (3, 3):
            return mat
        raise ValueError(f"Expected transform with shape [..., 3, 3] or [..., 4, 4], got {tuple(mat.shape)}")

    def gen_grid(
        self,
        input_bev: torch.Tensor,
        key_to_adj: torch.Tensor,
        *,
        bda: torch.Tensor | None = None,
        bda_adj: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate sampling grid that maps key-frame BEV to adjacent frame BEV."""

        batch_size, _, height, width = input_bev.shape
        base_grid = self._base_grid(height, width, input_bev.device, input_bev.dtype)
        grid = base_grid.view(1, height, width, 3, 1).expand(batch_size, -1, -1, -1, -1)

        key_to_adj = self._planar_transform(key_to_adj).to(dtype=input_bev.dtype, device=input_bev.device)
        if bda is not None:
            bda_curr = self._planar_transform(bda).to(dtype=input_bev.dtype, device=input_bev.device)
            if bda_adj is None:
                bda_adj = bda
            bda_prev = self._planar_transform(bda_adj).to(dtype=input_bev.dtype, device=input_bev.device)
            key_to_adj = bda_curr @ key_to_adj @ torch.inverse(bda_prev)

        feat2bev = self._feature_to_bev_transform(width, height, input_bev.device, input_bev.dtype).view(1, 3, 3)
        tf = torch.inverse(feat2bev).matmul(key_to_adj).matmul(feat2bev)
        warped = tf.view(batch_size, 1, 1, 3, 3).matmul(grid)
        normalize = torch.tensor([width - 1.0, height - 1.0], dtype=input_bev.dtype, device=input_bev.device)
        out_grid = warped[..., :2, 0] / normalize.view(1, 1, 1, 2) * 2.0 - 1.0
        return out_grid

    def shift_feature(
        self,
        input_bev: torch.Tensor,
        key_to_adj: torch.Tensor,
        *,
        bda: torch.Tensor | None = None,
        bda_adj: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Align adjacent BEV feature to key frame using bilinear grid sampling."""

        grid = self.gen_grid(input_bev, key_to_adj, bda=bda, bda_adj=bda_adj)
        return F.grid_sample(input_bev, grid.to(input_bev.dtype), align_corners=True)

    def align_history(
        self,
        bev_seq: torch.Tensor,
        *,
        history_to_key: torch.Tensor | None = None,
        bda: torch.Tensor | None = None,
        bda_adj: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if history_to_key is None:
            return bev_seq
        if history_to_key.dim() != 4:
            raise ValueError(
                "Expected history_to_key with shape [B, T, 3, 3] or [B, T, 4, 4], "
                f"got {tuple(history_to_key.shape)}"
            )

        aligned = [bev_seq[:, 0]]
        hist = bev_seq.shape[1]
        for step in range(1, hist):
            # key->adjacent transform mirrors BEVDet4D alignment path.
            key_to_adj = torch.inverse(history_to_key[:, step].to(bev_seq))
            warped = self.shift_feature(bev_seq[:, step], key_to_adj, bda=bda, bda_adj=bda_adj)
            aligned.append(warped)
        return torch.stack(aligned, dim=1)

    def forward(
        self,
        bev_seq: torch.Tensor,
        *,
        history_to_key: torch.Tensor | None = None,
        bda: torch.Tensor | None = None,
        bda_adj: torch.Tensor | None = None,
        return_aligned: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if bev_seq.dim() != 5:
            raise ValueError(f"Expected BEV sequence [B, T, C, H, W], got {tuple(bev_seq.shape)}")

        aligned_seq = bev_seq
        warp_cfg = self.cfg.temporal_warp
        if (
            warp_cfg.enabled
            and warp_cfg.align_after_view_transformation
            and history_to_key is not None
        ):
            aligned_seq = self.align_history(
                bev_seq,
                history_to_key=history_to_key,
                bda=bda,
                bda_adj=bda_adj,
            )

        batch_size, hist, embed_dims, bev_h, bev_w = aligned_seq.shape
        tokens = aligned_seq.permute(0, 3, 4, 2, 1).reshape(batch_size * bev_h * bev_w, embed_dims, hist)
        tokens = self.temporal_conv(tokens)
        tokens = self.norm(tokens)
        tokens = F.gelu(tokens)
        last_step = tokens[..., -1].view(batch_size, bev_h, bev_w, embed_dims).permute(0, 3, 1, 2)
        fused = self.proj(last_step)
        if return_aligned:
            return fused, tokens, aligned_seq
        return fused, tokens

