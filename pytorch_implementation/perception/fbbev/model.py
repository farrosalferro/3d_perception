"""Standalone FB-BEV forward-only model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .backbone_neck import BackboneNeck
from .backward_projection import BackwardProjectionLite
from .bev_encoder import BEVEncoderLite
from .config import FBBEVForwardConfig
from .depth_net import FBBEVDepthNetLite
from .detection_head import FBBEVDetectionHeadLite
from .forward_projection import ForwardProjectionLite
from .temporal_fusion import TemporalFusionLite


class FBBEVLite(nn.Module):
    """Pure-PyTorch FB-BEV implementation for forward-path study/testing."""

    def __init__(self, cfg: FBBEVForwardConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone_neck = BackboneNeck(cfg.backbone_neck)
        self.depth_net = FBBEVDepthNetLite(cfg)
        self.forward_projection = ForwardProjectionLite(cfg)
        self.backward_projection = BackwardProjectionLite(cfg)
        self.temporal_fusion = TemporalFusionLite(cfg)
        self.bev_encoder = BEVEncoderLite(cfg)
        self.detection_head = FBBEVDetectionHeadLite(cfg)

    def _shape_hw(
        self,
        shape: object,
        *,
        field_name: str,
        batch_idx: int,
        cam_idx: int,
    ) -> tuple[int, int]:
        if not isinstance(shape, (list, tuple)) or len(shape) < 2:
            raise ValueError(
                f"img_metas[{batch_idx}]['{field_name}'][{cam_idx}] must be a sequence with at least (H, W), got {shape!r}"
            )
        h, w = int(shape[0]), int(shape[1])
        if h <= 0 or w <= 0:
            raise ValueError(
                f"img_metas[{batch_idx}]['{field_name}'][{cam_idx}] has non-positive shape {(h, w)}"
            )
        return h, w

    def _validate_img_metas(self, img_metas: list[dict[str, Any]], *, batch_size: int, num_cams: int) -> None:
        if not isinstance(img_metas, list):
            raise TypeError(f"img_metas must be list[dict], got {type(img_metas)}")
        if len(img_metas) != batch_size:
            raise ValueError(f"Expected {batch_size} img_metas entries, got {len(img_metas)}")

        for batch_idx, meta in enumerate(img_metas):
            if not isinstance(meta, dict):
                raise TypeError(f"img_metas[{batch_idx}] must be dict, got {type(meta)}")

            if self.cfg.use_temporal_fusion:
                for key in ("sequence_group_idx", "start_of_sequence", "curr_to_prev_ego_rt"):
                    if key not in meta:
                        raise KeyError(f"img_metas[{batch_idx}] missing required temporal key '{key}'")
                curr_to_prev = torch.as_tensor(meta["curr_to_prev_ego_rt"])
                if curr_to_prev.shape != (4, 4):
                    raise ValueError(
                        f"img_metas[{batch_idx}]['curr_to_prev_ego_rt'] must be 4x4, got {tuple(curr_to_prev.shape)}"
                    )
                if not torch.isfinite(curr_to_prev).all():
                    raise ValueError(f"img_metas[{batch_idx}]['curr_to_prev_ego_rt'] must be finite.")

            if self.cfg.strict_img_meta:
                if "img_shape" not in meta:
                    raise KeyError(f"img_metas[{batch_idx}] must include 'img_shape'.")
                if "pad_shape" not in meta:
                    raise KeyError(f"img_metas[{batch_idx}] must include 'pad_shape'.")

            img_shapes = meta.get("img_shape")
            pad_shapes = meta.get("pad_shape", img_shapes)
            if img_shapes is not None:
                if not isinstance(img_shapes, (list, tuple)) or len(img_shapes) != num_cams:
                    raise ValueError(
                        f"img_metas[{batch_idx}]['img_shape'] must have {num_cams} camera entries, got {img_shapes!r}"
                    )
                if pad_shapes is None:
                    pad_shapes = img_shapes
                if not isinstance(pad_shapes, (list, tuple)) or len(pad_shapes) != num_cams:
                    raise ValueError(
                        f"img_metas[{batch_idx}]['pad_shape'] must have {num_cams} camera entries, got {pad_shapes!r}"
                    )
                for cam_idx in range(num_cams):
                    img_h, img_w = self._shape_hw(
                        img_shapes[cam_idx],
                        field_name="img_shape",
                        batch_idx=batch_idx,
                        cam_idx=cam_idx,
                    )
                    pad_h, pad_w = self._shape_hw(
                        pad_shapes[cam_idx],
                        field_name="pad_shape",
                        batch_idx=batch_idx,
                        cam_idx=cam_idx,
                    )
                    if pad_h < img_h or pad_w < img_w:
                        raise ValueError(
                            f"pad_shape must be >= img_shape for batch {batch_idx}, cam {cam_idx}; "
                            f"got pad={(pad_h, pad_w)} img={(img_h, img_w)}"
                        )

            lidar2img = meta.get("lidar2img")
            if lidar2img is not None:
                if not isinstance(lidar2img, (list, tuple)) or len(lidar2img) != num_cams:
                    raise ValueError(
                        f"img_metas[{batch_idx}]['lidar2img'] must have {num_cams} matrices, got {lidar2img!r}"
                    )
                for cam_idx, matrix in enumerate(lidar2img):
                    cam_mat = torch.as_tensor(matrix)
                    if cam_mat.shape != (4, 4):
                        raise ValueError(
                            f"img_metas[{batch_idx}]['lidar2img'][{cam_idx}] must be 4x4, got {tuple(cam_mat.shape)}"
                        )
                    if not torch.isfinite(cam_mat).all():
                        raise ValueError(
                            f"img_metas[{batch_idx}]['lidar2img'][{cam_idx}] contains non-finite values."
                        )

    def _extract_lidar2img(
        self,
        img_metas: list[dict[str, Any]],
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_cams: int,
    ) -> torch.Tensor | None:
        if not img_metas or any("lidar2img" not in meta for meta in img_metas):
            return None
        mats = []
        for batch_idx, meta in enumerate(img_metas):
            per_cam = []
            for cam_idx, matrix in enumerate(meta["lidar2img"]):
                cam_mat = torch.as_tensor(matrix, dtype=dtype, device=device)
                if cam_mat.shape != (4, 4):
                    raise ValueError(
                        f"img_metas[{batch_idx}]['lidar2img'][{cam_idx}] must be 4x4, got {tuple(cam_mat.shape)}"
                    )
                per_cam.append(cam_mat)
            if len(per_cam) != num_cams:
                raise ValueError(
                    f"img_metas[{batch_idx}]['lidar2img'] expected {num_cams} matrices, got {len(per_cam)}"
                )
            mats.append(torch.stack(per_cam, dim=0))
        return torch.stack(mats, dim=0)

    def _extract_cam_params(
        self,
        img_metas: list[dict[str, Any]],
        *,
        device: torch.device,
        dtype: torch.dtype,
        num_cams: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if not img_metas:
            return None

        rots_batch: list[torch.Tensor] = []
        trans_batch: list[torch.Tensor] = []
        intrins_batch: list[torch.Tensor] = []
        post_rots_batch: list[torch.Tensor] = []
        post_trans_batch: list[torch.Tensor] = []
        bda_batch: list[torch.Tensor] = []

        for batch_idx, meta in enumerate(img_metas):
            cam_params = meta.get("cam_params")
            if cam_params is None:
                required = ("rots", "trans", "intrins", "post_rots", "post_trans")
                if not all(key in meta for key in required):
                    return None
                cam_params = (
                    meta["rots"],
                    meta["trans"],
                    meta["intrins"],
                    meta["post_rots"],
                    meta["post_trans"],
                    meta.get("bda", torch.eye(3)),
                )

            if isinstance(cam_params, dict):
                required = ("rots", "trans", "intrins", "post_rots", "post_trans")
                if not all(key in cam_params for key in required):
                    raise KeyError(
                        f"img_metas[{batch_idx}]['cam_params'] must contain {required} and optional 'bda'."
                    )
                rots = cam_params["rots"]
                trans = cam_params["trans"]
                intrins = cam_params["intrins"]
                post_rots = cam_params["post_rots"]
                post_trans = cam_params["post_trans"]
                bda = cam_params.get("bda", torch.eye(3))
            else:
                if not isinstance(cam_params, (list, tuple)) or len(cam_params) != 6:
                    raise ValueError(
                        f"img_metas[{batch_idx}]['cam_params'] must be a 6-tuple/list, got {type(cam_params)}."
                    )
                rots, trans, intrins, post_rots, post_trans, bda = cam_params

            rots_t = torch.as_tensor(rots, dtype=dtype, device=device)
            trans_t = torch.as_tensor(trans, dtype=dtype, device=device)
            intrins_t = torch.as_tensor(intrins, dtype=dtype, device=device)
            post_rots_t = torch.as_tensor(post_rots, dtype=dtype, device=device)
            post_trans_t = torch.as_tensor(post_trans, dtype=dtype, device=device)
            bda_t = torch.as_tensor(bda, dtype=dtype, device=device)
            if bda_t.shape == (4, 4):
                bda_t = bda_t[:3, :3]

            if rots_t.shape != (num_cams, 3, 3):
                raise ValueError(f"img_metas[{batch_idx}] rots must be [{num_cams}, 3, 3], got {tuple(rots_t.shape)}")
            if trans_t.shape != (num_cams, 3):
                raise ValueError(f"img_metas[{batch_idx}] trans must be [{num_cams}, 3], got {tuple(trans_t.shape)}")
            if intrins_t.shape != (num_cams, 3, 3):
                raise ValueError(
                    f"img_metas[{batch_idx}] intrins must be [{num_cams}, 3, 3], got {tuple(intrins_t.shape)}"
                )
            if post_rots_t.shape != (num_cams, 3, 3):
                raise ValueError(
                    f"img_metas[{batch_idx}] post_rots must be [{num_cams}, 3, 3], got {tuple(post_rots_t.shape)}"
                )
            if post_trans_t.shape != (num_cams, 3):
                raise ValueError(
                    f"img_metas[{batch_idx}] post_trans must be [{num_cams}, 3], got {tuple(post_trans_t.shape)}"
                )
            if bda_t.shape != (3, 3):
                raise ValueError(f"img_metas[{batch_idx}] bda must be 3x3 or 4x4, got {tuple(bda_t.shape)}")

            rots_batch.append(rots_t)
            trans_batch.append(trans_t)
            intrins_batch.append(intrins_t)
            post_rots_batch.append(post_rots_t)
            post_trans_batch.append(post_trans_t)
            bda_batch.append(bda_t)

        return (
            torch.stack(rots_batch, dim=0),
            torch.stack(trans_batch, dim=0),
            torch.stack(intrins_batch, dim=0),
            torch.stack(post_rots_batch, dim=0),
            torch.stack(post_trans_batch, dim=0),
            torch.stack(bda_batch, dim=0),
        )

    def extract_img_feat(self, img: torch.Tensor) -> list[torch.Tensor]:
        """Extract multiscale camera features from [B, Ncam, 3, H, W]."""

        if img.dim() != 5:
            raise ValueError(f"Expected image shape [B, Ncam, 3, H, W], got {tuple(img.shape)}")
        batch_size, num_cams, channels, height, width = img.shape
        if num_cams != self.cfg.num_cams:
            raise ValueError(f"Expected num_cams={self.cfg.num_cams}, got {num_cams}")
        img_flat = img.reshape(batch_size * num_cams, channels, height, width)
        feats = self.backbone_neck(img_flat)
        return [feat.view(batch_size, num_cams, feat.shape[1], feat.shape[2], feat.shape[3]) for feat in feats]

    def _fuse_history(self, bev_refined: torch.Tensor, img_metas: list[dict]) -> torch.Tensor:
        if not self.cfg.use_temporal_fusion:
            return bev_refined
        return self.temporal_fusion(bev_refined, img_metas)

    def forward(
        self,
        img: torch.Tensor,
        img_metas: list[dict[str, Any]],
        *,
        decode: bool = False,
    ) -> dict[str, torch.Tensor] | dict[str, object]:
        batch_size, num_cams = img.shape[:2]
        self._validate_img_metas(img_metas, batch_size=batch_size, num_cams=num_cams)
        img_feats = self.extract_img_feat(img)
        camera_feat = img_feats[0]
        context, depth = self.depth_net(camera_feat)
        cam_params = self._extract_cam_params(
            img_metas,
            device=context.device,
            dtype=context.dtype,
            num_cams=num_cams,
        )
        lidar2img = self._extract_lidar2img(
            img_metas,
            device=context.device,
            dtype=context.dtype,
            num_cams=num_cams,
        )
        if self.cfg.require_camera_meta and cam_params is None and lidar2img is None:
            raise KeyError(
                "Camera metadata required but missing. Provide either 'cam_params' or 'lidar2img' in img_metas."
            )

        bev_volume = self.forward_projection(context, depth, cam_params=cam_params, img_metas=img_metas)
        bev_refined = self.backward_projection(
            bev_volume,
            context,
            depth,
            cam_params=cam_params,
            lidar2img=lidar2img,
            img_metas=img_metas,
        )
        bev_fused = self._fuse_history(bev_refined, img_metas)
        bev_embed = self.bev_encoder(bev_fused)
        outputs = self.detection_head(bev_embed, bev_volume=bev_volume)
        outputs.update(
            {
                "context": context,
                "depth": depth,
                "bev_volume": bev_volume,
                "bev_refined": bev_refined,
                "bev_fused": bev_fused,
                "bev_embed": bev_embed,
            }
        )
        if decode:
            return {
                "preds": outputs,
                "decoded": self.detection_head.get_bboxes(outputs),
                "decoded_occupancy": self.detection_head.decode_occupancy(
                    outputs,
                    fix_void=self.cfg.occupancy_fix_void,
                    return_raw_occ=False,
                ),
            }
        return outputs

    def simple_test(
        self,
        img: torch.Tensor,
        img_metas: list[dict[str, Any]],
        *,
        return_raw_occ: bool = False,
    ) -> list[dict[str, object]]:
        outputs = self.forward(img, img_metas, decode=False)
        if not isinstance(outputs, dict):
            raise TypeError("Expected forward outputs to be a dict.")
        det = self.detection_head.get_bboxes(outputs)
        occ = self.detection_head.decode_occupancy(
            outputs,
            fix_void=self.cfg.occupancy_fix_void,
            return_raw_occ=return_raw_occ,
        )
        results: list[dict[str, object]] = []
        for batch_idx, (det_item, occ_item) in enumerate(zip(det, occ)):
            sample = {"pts_bbox": det_item, "pred_occupancy": occ_item}
            meta = img_metas[batch_idx]
            if "index" in meta:
                sample["index"] = meta["index"]
            results.append(sample)
        return results

    def clear_temporal_state(self) -> None:
        self.temporal_fusion.clear()
