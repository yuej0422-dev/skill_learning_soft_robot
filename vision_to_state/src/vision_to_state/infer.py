from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from PIL import Image

from .data import STATE_NAMES
from .model import build_model


def load_rgbd_pair(rgb_path: str, depth_path: str, data_cfg: dict) -> torch.Tensor:
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    depth = np.asarray(Image.open(depth_path))
    if depth.ndim == 3:
        depth = depth[..., 0]
    h, w = rgb.shape[:2]
    x0 = int(round(w * float(data_cfg["crop_left_frac"])))
    y0 = int(round(h * float(data_cfg["crop_top_frac"])))
    rgb = rgb[y0:, x0:, :]
    depth = depth[y0:, x0:]
    size = int(data_cfg["image_size"])
    rgb = np.asarray(Image.fromarray(rgb).resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    depth = np.asarray(Image.fromarray(depth).resize((size, size), Image.NEAREST), dtype=np.float32)
    depth = np.clip(depth, 0.0, float(data_cfg["depth_clip_mm"])) / float(data_cfg["depth_clip_mm"])
    return torch.from_numpy(np.concatenate([np.transpose(rgb, (2, 0, 1)), depth[None]], axis=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rgb", nargs="+", required=True)
    parser.add_argument("--depth", nargs="+", required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    if len(args.rgb) != len(args.depth):
        raise ValueError("--rgb and --depth must have the same number of frames.")
    if len(args.rgb) != int(cfg["data"]["seq_len"]):
        raise ValueError(f"Expected {cfg['data']['seq_len']} frames.")

    model = build_model(cfg["model"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    seq = torch.stack(
        [load_rgbd_pair(r, d, cfg["data"]) for r, d in zip(args.rgb, args.depth)],
        dim=0,
    )[None]
    with torch.no_grad():
        pred_norm = model(seq)[0].numpy()
    mean = np.asarray(ckpt["normalizer"]["mean"], dtype=np.float32)
    std = np.asarray(ckpt["normalizer"]["std"], dtype=np.float32)
    pred = pred_norm * std + mean
    print(json.dumps(dict(zip(STATE_NAMES, pred.tolist())), indent=2))


if __name__ == "__main__":
    main()
