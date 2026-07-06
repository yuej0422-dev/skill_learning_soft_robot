from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import RGBDStateDataset, load_or_build_manifest, pretty_metrics, split_records
from .model import build_model
from .train import evaluate, make_loader, select_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    manifest = load_or_build_manifest(cfg, rebuild=False)
    splits = split_records(
        [r for r in manifest["records"] if r["status"] == "keep"],
        train_ratio=float(cfg["data"]["train_ratio"]),
        val_ratio=float(cfg["data"]["val_ratio"]),
        seed=int(cfg["data"]["split_seed"]),
    )
    dataset = RGBDStateDataset(
        manifest["source_zip"],
        splits[args.split],
        cfg["data"],
        normalizer=ckpt["normalizer"],
    )
    loader = make_loader(dataset, args.batch_size, args.num_workers, shuffle=False)
    device = select_device(args.device)
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    metrics = evaluate(model, loader, device, ckpt["normalizer"])

    payload = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "samples": len(dataset),
        "metrics": metrics,
    }
    print(f"{args.split} samples={len(dataset)} {pretty_metrics(metrics)}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
