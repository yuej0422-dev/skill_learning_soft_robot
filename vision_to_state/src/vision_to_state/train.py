from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config, with_overrides
from .data import (
    RGBDStateDataset,
    compute_normalizer,
    load_or_build_manifest,
    pretty_metrics,
    rmse_mae,
    split_records,
)
from .model import build_model


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_loader(dataset: RGBDStateDataset, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle and len(dataset) >= batch_size,
    )


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    normalizer: dict[str, list[float]],
) -> dict[str, Any]:
    model.eval()
    preds = []
    targets = []
    mean = torch.tensor(normalizer["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(normalizer["std"], dtype=torch.float32, device=device)
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y_raw = batch["raw_target"].numpy()
            pred_norm = model(x)
            pred_raw = (pred_norm * std + mean).detach().cpu().numpy()
            preds.append(pred_raw)
            targets.append(y_raw)
    if not preds:
        return {"pose_rmse_mean": float("nan"), "vel_rmse_mean": float("nan")}
    return rmse_mae(np.concatenate(preds, axis=0), np.concatenate(targets, axis=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = with_overrides(
        cfg,
        {
            "train.epochs": args.epochs,
            "train.batch_size": args.batch_size,
            "train.max_records": args.max_records,
            "train.max_samples": args.max_samples,
            "train.run_dir": args.run_dir,
        },
    )

    manifest = load_or_build_manifest(cfg, rebuild=args.rebuild_manifest)
    records = [r for r in manifest["records"] if r["status"] == "keep"]
    if cfg["train"].get("max_records") is not None:
        records = records[: int(cfg["train"]["max_records"])]
    splits = split_records(
        records,
        train_ratio=float(cfg["data"]["train_ratio"]),
        val_ratio=float(cfg["data"]["val_ratio"]),
        seed=int(cfg["data"]["split_seed"]),
    )
    if not splits["train"]:
        raise RuntimeError("No valid training records after data-quality filtering.")

    max_samples = cfg["train"].get("max_samples")
    train_ds_for_stats = RGBDStateDataset(
        manifest["source_zip"],
        splits["train"],
        cfg["data"],
        normalizer=None,
        max_samples=max_samples,
    )
    normalizer = compute_normalizer(train_ds_for_stats)
    train_ds = RGBDStateDataset(
        manifest["source_zip"],
        splits["train"],
        cfg["data"],
        normalizer=normalizer,
        max_samples=max_samples,
    )
    val_ds = RGBDStateDataset(
        manifest["source_zip"],
        splits["val"] or splits["train"][:1],
        cfg["data"],
        normalizer=normalizer,
        max_samples=max_samples,
    )

    device = select_device(str(cfg["train"]["device"]))
    model = build_model(cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_weights = torch.tensor(
        [float(cfg["loss"]["pose_weight"])] * 6 + [float(cfg["loss"]["velocity_weight"])] * 6,
        dtype=torch.float32,
        device=device,
    )

    train_loader = make_loader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        workers=int(cfg["train"]["num_workers"]),
        shuffle=True,
    )
    val_loader = make_loader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        workers=int(cfg["train"]["num_workers"]),
        shuffle=False,
    )

    run_dir = Path(cfg["train"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    with open(run_dir / "normalizer.json", "w", encoding="utf-8") as f:
        json.dump(normalizer, f, indent=2)

    print(f"device={device} train_samples={len(train_ds)} val_samples={len(val_ds)}")
    print(f"records train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    best_pose = float("inf")
    history = []
    try:
        for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
            model.train()
            losses = []
            pbar = tqdm(train_loader, desc=f"epoch {epoch}", dynamic_ncols=True)
            for batch in pbar:
                x = batch["image"].to(device, non_blocking=True)
                y = batch["target"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(x)
                    loss = ((pred - y).pow(2) * loss_weights).mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu()))
                pbar.set_postfix(loss=f"{losses[-1]:.5f}")

            val_metrics = evaluate(model, val_loader, device, normalizer)
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "val": val_metrics,
            }
            history.append(row)
            print(
                f"epoch={epoch} train_loss={row['train_loss']:.6f} "
                f"val {pretty_metrics(val_metrics)}"
            )

            ckpt = {
                "model": model.state_dict(),
                "config": cfg,
                "normalizer": normalizer,
                "epoch": epoch,
                "val_metrics": val_metrics,
            }
            if val_metrics["pose_rmse_mean"] < best_pose:
                best_pose = val_metrics["pose_rmse_mean"]
                torch.save(ckpt, run_dir / "best.pt")
            if epoch % int(cfg["train"]["save_every_epochs"]) == 0:
                torch.save(ckpt, run_dir / f"epoch_{epoch:04d}.pt")

            with open(run_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
    except KeyboardInterrupt:
        print("Training interrupted; completed epoch history has been kept.")

    print(f"best checkpoint: {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
