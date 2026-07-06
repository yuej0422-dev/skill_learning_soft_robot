from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .data import STATE_NAMES, pretty_metrics, rmse_mae
from .model import build_model


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def read_parquet_tree(path: Path) -> dict[str, Any]:
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {path}")
    tables = [pq.read_table(file) for file in files]
    return pa.concat_tables(tables).to_pydict()


def load_state_quantile_normalizer(root: Path, target_min: float, target_max: float) -> dict[str, Any]:
    stats = json.loads((root / "meta" / "stats.json").read_text(encoding="utf-8"))
    state_stats = stats["observation.state"]
    q01 = np.asarray(state_stats["q01"], dtype=np.float32)
    q99 = np.asarray(state_stats["q99"], dtype=np.float32)
    scale = q99 - q01
    scale = np.where(np.abs(scale) < 1e-8, 1.0, scale).astype(np.float32)
    return {
        "type": "q01_q99_minmax",
        "q01": q01.tolist(),
        "q99": q99.tolist(),
        "scale": scale.tolist(),
        "target_min": float(target_min),
        "target_max": float(target_max),
        "clip_target_to_range": True,
        "names": STATE_NAMES,
    }


def normalize_state(state: np.ndarray, normalizer: dict[str, Any]) -> np.ndarray:
    q01 = np.asarray(normalizer["q01"], dtype=np.float32)
    scale = np.asarray(normalizer["scale"], dtype=np.float32)
    target_min = float(normalizer["target_min"])
    target_max = float(normalizer["target_max"])
    unit = (state.astype(np.float32) - q01) / scale
    unit = np.clip(unit, 0.0, 1.0)
    out = unit * (target_max - target_min) + target_min
    return out.astype(np.float32)


def denormalize_state(state_norm: np.ndarray, normalizer: dict[str, Any]) -> np.ndarray:
    q01 = np.asarray(normalizer["q01"], dtype=np.float32)
    scale = np.asarray(normalizer["scale"], dtype=np.float32)
    target_min = float(normalizer["target_min"])
    target_max = float(normalizer["target_max"])
    unit = (state_norm.astype(np.float32) - target_min) / (target_max - target_min)
    return unit * scale + q01


def preprocess_rgb(
    rgb: np.ndarray,
    image_size: int,
    crop_left_frac: float,
    crop_top_frac: float,
) -> np.ndarray:
    h, w = rgb.shape[:2]
    x0 = int(round(w * crop_left_frac))
    y0 = int(round(h * crop_top_frac))
    img = Image.fromarray(rgb[y0:, x0:, :], mode="RGB")
    img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.transpose(np.asarray(img, dtype=np.uint8), (2, 0, 1))


def preprocess_depth(
    depth: np.ndarray,
    image_size: int,
    crop_left_frac: float,
    crop_top_frac: float,
    depth_clip_mm: float,
) -> np.ndarray:
    h, w = depth.shape[:2]
    x0 = int(round(w * crop_left_frac))
    y0 = int(round(h * crop_top_frac))
    img = Image.fromarray(depth[y0:, x0:])
    img = img.resize((image_size, image_size), Image.Resampling.NEAREST)
    arr = np.asarray(img, dtype=np.float32)
    arr = np.clip(arr, 0.0, depth_clip_mm) / depth_clip_mm
    return np.round(arr * 255.0).astype(np.uint8)[None, :, :]


def preprocess_depth_batch(
    depths: np.ndarray,
    raw_indices: list[int],
    image_size: int,
    crop_left_frac: float,
    crop_top_frac: float,
    depth_clip_mm: float,
) -> np.ndarray:
    if not raw_indices:
        return np.empty((0, 1, image_size, image_size), dtype=np.uint8)
    h, w = depths.shape[1:3]
    x0 = int(round(w * crop_left_frac))
    y0 = int(round(h * crop_top_frac))
    cropped = depths[np.asarray(raw_indices, dtype=np.int64), y0:, x0:]
    ys = np.rint(np.linspace(0, cropped.shape[1] - 1, image_size)).astype(np.int64)
    xs = np.rint(np.linspace(0, cropped.shape[2] - 1, image_size)).astype(np.int64)
    resized = cropped[:, ys, :][:, :, xs].astype(np.float32)
    resized = np.clip(resized, 0.0, depth_clip_mm) / depth_clip_mm
    return np.round(resized * 255.0).astype(np.uint8)[:, None, :, :]


def find_camera_video(root: Path, camera: str) -> list[Path]:
    video_root = root / "videos" / f"observation.images.{camera}"
    files = sorted(video_root.rglob("*.mp4"))
    if not files:
        raise FileNotFoundError(f"No mp4 files found under {video_root}")
    return files


def build_rgb_cache(
    root: Path,
    camera: str,
    n_rows: int,
    image_size: int,
    crop_left_frac: float,
    crop_top_frac: float,
) -> np.ndarray:
    cache = np.empty((n_rows, 4, image_size, image_size), dtype=np.uint8)
    write_index = 0
    for video_path in find_camera_video(root, camera):
        with av.open(str(video_path)) as container:
            for frame in tqdm(
                container.decode(video=0),
                desc=f"decode {camera} {video_path.name}",
                dynamic_ncols=True,
            ):
                if write_index >= n_rows:
                    break
                rgb = frame.to_ndarray(format="rgb24")
                cache[write_index, :3] = preprocess_rgb(
                    rgb,
                    image_size=image_size,
                    crop_left_frac=crop_left_frac,
                    crop_top_frac=crop_top_frac,
                )
                write_index += 1
    if write_index != n_rows:
        raise RuntimeError(f"Decoded {write_index} RGB frames, expected {n_rows}")
    return cache


def build_depth_cache(
    root: Path,
    camera: str,
    data: dict[str, Any],
    cache: np.ndarray,
    image_size: int,
    crop_left_frac: float,
    crop_top_frac: float,
    depth_clip_mm: float,
) -> None:
    depth_index = read_parquet_tree(root / "meta" / "extra")
    row_for_key = {
        (int(data["episode_index"][row]), int(data["frame_index"][row])): row
        for row in range(len(data["index"]))
    }
    groups: dict[str, list[tuple[int, int]]] = {}
    for i, cam in enumerate(depth_index["camera_name"]):
        if cam != camera:
            continue
        key = (int(depth_index["episode_index"][i]), int(depth_index["frame_index"][i]))
        row = row_for_key.get(key)
        if row is None:
            continue
        rel_path = str(depth_index["raw_depth_path"][i])
        raw_index = int(depth_index["raw_array_index"][i])
        groups.setdefault(rel_path, []).append((row, raw_index))

    filled = np.zeros(len(data["index"]), dtype=bool)
    for rel_path, rows_and_indices in tqdm(
        sorted(groups.items()),
        desc=f"load {camera} depth episodes",
        dynamic_ncols=True,
    ):
        rows = [x[0] for x in rows_and_indices]
        raw_indices = [x[1] for x in rows_and_indices]
        with np.load(root / rel_path) as npz:
            cache[np.asarray(rows, dtype=np.int64), 3:4] = preprocess_depth_batch(
                npz["depth"],
                raw_indices,
                image_size=image_size,
                crop_left_frac=crop_left_frac,
                crop_top_frac=crop_top_frac,
                depth_clip_mm=depth_clip_mm,
            )
        filled[np.asarray(rows, dtype=np.int64)] = True
    if not filled.all():
        missing = np.flatnonzero(~filled)[:10].tolist()
        raise RuntimeError(f"Missing depth frames for {camera}; first missing row indices: {missing}")


def split_episodes(
    episode_indices: list[int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, set[int]]:
    episodes = sorted(set(int(x) for x in episode_indices))
    rng = random.Random(seed)
    rng.shuffle(episodes)
    n = len(episodes)
    n_train = min(n, int(round(n * train_ratio)))
    n_val = min(n - n_train, int(round(n * val_ratio)))
    return {
        "train": set(episodes[:n_train]),
        "val": set(episodes[n_train : n_train + n_val]),
        "test": set(episodes[n_train + n_val :]),
    }


def make_samples(data: dict[str, Any], episode_set: set[int], seq_len: int) -> list[int]:
    samples: list[int] = []
    episodes = [int(x) for x in data["episode_index"]]
    frame_indices = [int(x) for x in data["frame_index"]]
    for i, episode in enumerate(episodes):
        if episode not in episode_set:
            continue
        if frame_indices[i] < seq_len - 1:
            continue
        start = i - seq_len + 1
        if start < 0:
            continue
        if all(episodes[j] == episode for j in range(start, i + 1)):
            samples.append(i)
    return samples


class CachedLeRobotRGBDStateDataset(Dataset):
    def __init__(
        self,
        frame_cache: np.ndarray,
        states: np.ndarray,
        samples: list[int],
        normalizer: dict[str, Any],
        seq_len: int,
    ) -> None:
        self.frame_cache = frame_cache
        self.states = states.astype(np.float32)
        self.samples = samples
        self.normalizer = normalizer
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        target_index = self.samples[index]
        start = target_index - self.seq_len + 1
        image = self.frame_cache[start : target_index + 1].astype(np.float32) / 255.0
        raw_target = self.states[target_index]
        target = normalize_state(raw_target, self.normalizer)
        return {
            "image": torch.from_numpy(image),
            "target": torch.from_numpy(target),
            "raw_target": torch.from_numpy(raw_target),
        }


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    normalizer: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            pred_norm = model(x).detach().cpu().numpy()
            preds.append(denormalize_state(pred_norm, normalizer))
            targets.append(batch["raw_target"].numpy())
    if not preds:
        return {"pose_rmse_mean": float("nan"), "vel_rmse_mean": float("nan")}
    return rmse_mae(np.concatenate(preds, axis=0), np.concatenate(targets, axis=0))


def write_run_summary(
    path: Path,
    args: argparse.Namespace,
    splits: dict[str, set[int]],
    sample_counts: dict[str, int],
    best_metrics: dict[str, Any],
    best_epoch: int | None,
    best_mae_metrics: dict[str, Any],
    best_mae_epoch: int | None,
    test_metrics: dict[str, Any] | None,
) -> None:
    lines = [
        "# LeRobot RGB-D -> 12D State Training",
        "",
        f"- dataset: `{args.root}`",
        f"- camera: `{args.camera}`",
        f"- seq_len: `{args.seq_len}`",
        f"- image_size: `{args.image_size}`",
        f"- target normalization: `observation.state` q01/q99 min-max with target clipping to [{args.target_min:g}, {args.target_max:g}]",
        f"- loss weights: pose={args.pose_weight}, velocity={args.velocity_weight}",
        f"- train/val/test episodes: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}",
        f"- train/val/test samples: {sample_counts['train']}/{sample_counts['val']}/{sample_counts['test']}",
        f"- best val RMSE epoch: {best_epoch}",
        f"- best val pose RMSE: {best_metrics.get('pose_rmse_mean', float('nan')):.6f}",
        f"- best val MAE epoch: {best_mae_epoch}",
        f"- best val pose MAE: {best_mae_metrics.get('pose_mae_mean', float('nan')):.6f}",
        f"- best val velocity RMSE: {best_metrics.get('vel_rmse_mean', float('nan')):.6f} (velocity loss weight is zero)",
    ]
    if test_metrics is not None:
        lines.extend(
            [
                f"- test pose RMSE: {test_metrics.get('pose_rmse_mean', float('nan')):.6f}",
                f"- test pose MAE: {test_metrics.get('pose_mae_mean', float('nan')):.6f}",
                f"- test velocity RMSE: {test_metrics.get('vel_rmse_mean', float('nan')):.6f} (velocity loss weight is zero)",
            ]
        )
    lines.extend(
        [
            "",
            "Artifacts:",
            "",
            "- `best.pt`",
            "- `last.pt`",
            "- `history.json`",
            "- `normalizer_q01_q99.json`",
            "- `config_used.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="LeRobot v3 dataset root")
    parser.add_argument("--camera", default="cam_2")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--depth-clip-mm", type=float, default=1500.0)
    parser.add_argument("--crop-left-frac", type=float, default=0.25)
    parser.add_argument("--crop-top-frac", type=float, default=0.10)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-min", type=float, default=-1.0)
    parser.add_argument("--target-max", type=float, default=1.0)
    parser.add_argument("--pose-weight", type=float, default=1.0)
    parser.add_argument("--velocity-weight", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--frame-embed-dim", type=int, default=256)
    parser.add_argument("--temporal-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    args = parser.parse_args()

    root = Path(args.root)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    data = read_parquet_tree(root / "data")
    states = np.asarray(data["observation.state"], dtype=np.float32)
    normalizer = load_state_quantile_normalizer(root, args.target_min, args.target_max)

    frame_cache = build_rgb_cache(
        root=root,
        camera=args.camera,
        n_rows=len(data["index"]),
        image_size=args.image_size,
        crop_left_frac=args.crop_left_frac,
        crop_top_frac=args.crop_top_frac,
    )
    build_depth_cache(
        root=root,
        camera=args.camera,
        data=data,
        cache=frame_cache,
        image_size=args.image_size,
        crop_left_frac=args.crop_left_frac,
        crop_top_frac=args.crop_top_frac,
        depth_clip_mm=args.depth_clip_mm,
    )

    splits = split_episodes(
        data["episode_index"],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    samples = {name: make_samples(data, eps, args.seq_len) for name, eps in splits.items()}
    if not samples["train"]:
        raise RuntimeError("No training samples after episode split.")
    if not samples["val"]:
        samples["val"] = samples["train"][: min(256, len(samples["train"]))]

    datasets = {
        name: CachedLeRobotRGBDStateDataset(frame_cache, states, ids, normalizer, args.seq_len)
        for name, ids in samples.items()
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=len(datasets["train"]) >= args.batch_size,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }

    device = select_device(args.device)
    model = build_model(
        {
            "in_channels": 4,
            "frame_embed_dim": args.frame_embed_dim,
            "temporal_hidden_dim": args.temporal_hidden_dim,
            "dropout": args.dropout,
        }
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    weights = torch.tensor(
        [args.pose_weight] * 6 + [args.velocity_weight] * 6,
        dtype=torch.float32,
        device=device,
    )
    weight_denom = torch.clamp(weights.sum(), min=1e-8)

    config_used = vars(args).copy()
    config_used["normalizer"] = normalizer
    (run_dir / "config_used.json").write_text(json.dumps(config_used, indent=2), encoding="utf-8")
    (run_dir / "normalizer_q01_q99.json").write_text(
        json.dumps(normalizer, indent=2),
        encoding="utf-8",
    )

    print(
        f"device={device} rows={len(data['index'])} "
        f"samples train/val/test={len(samples['train'])}/{len(samples['val'])}/{len(samples['test'])}"
    )
    print(f"episodes train/val/test={len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")

    best_pose = float("inf")
    best_metrics: dict[str, Any] = {}
    best_epoch: int | None = None
    best_pose_mae = float("inf")
    best_mae_metrics: dict[str, Any] = {}
    best_mae_epoch: int | None = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(loaders["train"], desc=f"epoch {epoch}", dynamic_ncols=True)
        for batch in pbar:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(x)
                loss = ((pred - y).pow(2) * weights).sum(dim=1).mean() / weight_denom
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            pbar.set_postfix(loss=f"{losses[-1]:.5f}")

        val_metrics = evaluate(model, loaders["val"], device, normalizer)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "val": val_metrics,
        }
        history.append(row)
        print(f"epoch={epoch} train_loss={row['train_loss']:.6f} val {pretty_metrics(val_metrics)}")

        checkpoint = {
            "model": model.state_dict(),
            "normalizer": normalizer,
            "config": config_used,
            "epoch": epoch,
            "val_metrics": val_metrics,
        }
        if val_metrics["pose_rmse_mean"] < best_pose:
            best_pose = val_metrics["pose_rmse_mean"]
            best_metrics = val_metrics
            best_epoch = epoch
            torch.save(checkpoint, run_dir / "best.pt")
        if val_metrics["pose_mae_mean"] < best_pose_mae:
            best_pose_mae = val_metrics["pose_mae_mean"]
            best_mae_metrics = val_metrics
            best_mae_epoch = epoch
        torch.save(checkpoint, run_dir / "last.pt")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    test_metrics = None
    if samples["test"]:
        best_ckpt = torch.load(run_dir / "best.pt", map_location=device)
        model.load_state_dict(best_ckpt["model"])
        test_metrics = evaluate(model, loaders["test"], device, normalizer)
        (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
        print(f"test {pretty_metrics(test_metrics)}")

    write_run_summary(
        run_dir / "run_summary.md",
        args,
        splits,
        {name: len(ids) for name, ids in samples.items()},
        best_metrics,
        best_epoch,
        best_mae_metrics,
        best_mae_epoch,
        test_metrics,
    )
    print(f"best checkpoint: {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
