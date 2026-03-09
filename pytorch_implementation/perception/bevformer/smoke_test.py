"""Synthetic forward smoke tests for the pure PyTorch BEVFormer."""

from __future__ import annotations

import argparse
from typing import Any

import torch

from .config import base_forward_config, debug_forward_config, tiny_forward_config
from .model import BEVFormerLite


def _build_dummy_img_metas(
    batch_size: int,
    num_cams: int,
    height: int,
    width: int,
    *,
    can_bus_dims: int = 18,
) -> list[dict[str, Any]]:
    metas = []
    for batch_idx in range(batch_size):
        can_bus = [0.0] * can_bus_dims
        lidar2img = []
        for cam_idx in range(num_cams):
            projection = [
                [float(width), 0.0, 0.0, float(cam_idx) * 0.01 * width],
                [0.0, float(height), 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            lidar2img.append(projection)
        metas.append(
            {
                "scene_token": f"scene_{batch_idx}",
                "can_bus": can_bus,
                "lidar2img": lidar2img,
                "img_shape": [(height, width, 3) for _ in range(num_cams)],
            }
        )
    return metas


def run_smoke_test(
    *,
    config_name: str = "tiny",
    use_debug_config: bool = True,
    device: str = "cpu",
    batch_size: int = 1,
    height: int = 192,
    width: int = 320,
) -> dict[str, Any]:
    if config_name not in {"tiny", "base"}:
        raise ValueError("config_name must be 'tiny' or 'base'.")
    cfg = tiny_forward_config() if config_name == "tiny" else base_forward_config()
    if use_debug_config:
        cfg = debug_forward_config(config_name)

    model = BEVFormerLite(cfg).to(device).eval()
    img = torch.randn(batch_size, cfg.num_cams, 3, height, width, device=device)
    img_metas = _build_dummy_img_metas(
        batch_size=batch_size,
        num_cams=cfg.num_cams,
        height=height,
        width=width,
        can_bus_dims=cfg.can_bus_dims,
    )

    with torch.no_grad():
        outputs = model(img, img_metas, decode=True)

    preds = outputs["preds"]
    decoded = outputs["decoded"]
    cls_scores = preds["all_cls_scores"]
    bbox_preds = preds["all_bbox_preds"]
    bev_embed = preds["bev_embed"]

    expected_cls_shape = (cfg.num_decoder_layers, batch_size, cfg.num_queries, cfg.num_classes)
    expected_bbox_shape = (cfg.num_decoder_layers, batch_size, cfg.num_queries, 10)
    expected_bev_shape = (cfg.bev_h * cfg.bev_w, batch_size, cfg.embed_dims)

    assert cls_scores.shape == expected_cls_shape, (cls_scores.shape, expected_cls_shape)
    assert bbox_preds.shape == expected_bbox_shape, (bbox_preds.shape, expected_bbox_shape)
    assert bev_embed.shape == expected_bev_shape, (bev_embed.shape, expected_bev_shape)
    assert len(decoded) == batch_size

    return {
        "config_name": cfg.name,
        "cls_scores_shape": tuple(cls_scores.shape),
        "bbox_preds_shape": tuple(bbox_preds.shape),
        "bev_embed_shape": tuple(bev_embed.shape),
        "decoded_counts": [int(item["scores"].numel()) for item in decoded],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pure-PyTorch BEVFormer forward smoke test.")
    parser.add_argument("--config", choices=["tiny", "base"], default="tiny")
    parser.add_argument("--full-config", action="store_true", help="Use full tiny/base config instead of debug config.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    args = parser.parse_args()

    summary = run_smoke_test(
        config_name=args.config,
        use_debug_config=not args.full_config,
        device=args.device,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
    )
    print("Smoke test summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
