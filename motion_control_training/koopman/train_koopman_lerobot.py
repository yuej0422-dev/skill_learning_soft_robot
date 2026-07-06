from __future__ import annotations

import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - depends on runtime env
    raise SystemExit("pyarrow is required to read LeRobot parquet files. Use the soft_vla_cuda conda env.") from exc

try:
    from .model import KoopmanLossWeights, KoopmanNetwork, define_koopman_loss
except ImportError:  # pragma: no cover - direct script execution
    from model import KoopmanLossWeights, KoopmanNetwork, define_koopman_loss


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "lerobot_conversion/outputs/robot_records_7_03_1_delta_tcp"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "runs"


def parse_int_list(spec: str, total_dim: int | None = None) -> list[int]:
    spec = spec.strip()
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid slice spec: {spec}")
        start = int(parts[0]) if parts[0] else 0
        if parts[1]:
            stop = int(parts[1])
        elif total_dim is not None:
            stop = total_dim
        else:
            raise ValueError(f"Open-ended slice needs total_dim: {spec}")
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        return list(range(start, stop, step))
    return [int(item) for item in spec.split(",") if item.strip()]


def parse_hidden_sizes(spec: str) -> list[int]:
    return [int(item) for item in spec.split(",") if item.strip()]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parquet_files(dataset_root: Path) -> list[Path]:
    files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")
    return files


def load_lerobot_state_stats(dataset_root: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    stats = load_json(dataset_root / "meta/stats.json")
    info = load_json(dataset_root / "meta/info.json")
    state_stats = stats["observation.state"]
    state_feature = info["features"]["observation.state"]
    mean = np.asarray(state_stats["mean"], dtype=np.float32)
    std = np.asarray(state_stats["std"], dtype=np.float32)
    names = list(state_feature.get("names") or [f"state_{i}" for i in range(len(mean))])
    if mean.shape != std.shape:
        raise ValueError(f"State mean/std shape mismatch: {mean.shape} vs {std.shape}")
    return mean, std, names


def load_pressure_metadata(dataset_root: Path) -> tuple[list[str], dict[int, str]]:
    metadata = load_json(dataset_root / "meta/extra/raw_pressure_metadata.json")
    columns = list(metadata["columns"])
    episode_to_path = {
        int(item["episode_index"]): item["raw_pressure_path"]
        for item in metadata["episodes"]
    }
    return columns, episode_to_path


def load_episode_arrays(
    dataset_root: Path,
    state_indices: Sequence[int],
    pressure_indices: Sequence[int],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    rows: list[tuple[int, int, np.ndarray]] = []
    for path in parquet_files(dataset_root):
        table = pq.read_table(path, columns=["observation.state", "episode_index", "frame_index"])
        states = table["observation.state"].to_pylist()
        episodes = table["episode_index"].to_pylist()
        frames = table["frame_index"].to_pylist()
        for state, episode, frame in zip(states, episodes, frames):
            rows.append((int(episode), int(frame), np.asarray(state, dtype=np.float32)[state_indices]))

    by_episode: dict[int, list[tuple[int, np.ndarray]]] = {}
    for episode, frame, state in rows:
        by_episode.setdefault(episode, []).append((frame, state))

    pressure_columns, episode_to_path = load_pressure_metadata(dataset_root)
    if max(pressure_indices) >= len(pressure_columns):
        raise ValueError(f"Pressure index out of range for {len(pressure_columns)} columns: {pressure_indices}")

    episodes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for episode, frame_states in sorted(by_episode.items()):
        frame_states.sort(key=lambda item: item[0])
        frames = np.asarray([item[0] for item in frame_states], dtype=np.int64)
        states = np.stack([item[1] for item in frame_states], axis=0).astype(np.float32)
        expected = np.arange(len(frames), dtype=np.int64)
        if not np.array_equal(frames, expected):
            raise ValueError(f"Episode {episode} frame_index is not contiguous from 0.")

        pressure_path = dataset_root / episode_to_path[int(episode)]
        raw_pressure = np.load(pressure_path).astype(np.float32)
        if frames[-1] >= raw_pressure.shape[0]:
            raise ValueError(
                f"Episode {episode} has frame {frames[-1]} but pressure file has {raw_pressure.shape[0]} frames."
            )
        pressures = raw_pressure[frames][:, pressure_indices].astype(np.float32)
        episodes[int(episode)] = (states, pressures)
    return episodes


def split_episodes(episodes: Sequence[int], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    episode_ids = np.asarray(list(episodes), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(episode_ids)
    val_count = max(1, int(round(len(episode_ids) * val_ratio)))
    val_episodes = sorted(int(v) for v in episode_ids[:val_count])
    train_episodes = sorted(int(v) for v in episode_ids[val_count:])
    return train_episodes, val_episodes


def build_koopman_buffer(
    episodes: dict[int, tuple[np.ndarray, np.ndarray]],
    episode_ids: Sequence[int],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    ksteps: int,
) -> tuple[np.ndarray, dict[str, int]]:
    windows: list[np.ndarray] = []
    skipped = 0
    for episode in episode_ids:
        states, pressures = episodes[int(episode)]
        if len(states) <= ksteps:
            skipped += 1
            continue
        norm_states = (states - state_mean) / state_std
        data = np.concatenate([pressures, norm_states], axis=1).astype(np.float32)
        for start in range(0, len(data) - ksteps):
            windows.append(data[start : start + ksteps + 1])
    if not windows:
        raise ValueError(f"No training windows built. Reduce --ksteps; current value is {ksteps}.")
    buffer = np.stack(windows, axis=0).astype(np.float32)
    stats = {
        "episodes": len(episode_ids),
        "skipped_short_episodes": skipped,
        "windows": int(buffer.shape[0]),
    }
    return buffer, stats


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def make_output_dir(output_root: Path, run_name: str | None) -> Path:
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def init_wandb(args: argparse.Namespace, output_dir: Path, config: dict, metadata: dict):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise SystemExit("wandb is not installed in this environment. Install wandb or omit --wandb.") from exc

    tags = [tag for tag in args.wandb_tags.split(",") if tag.strip()]
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_name or output_dir.name,
        dir=str(output_dir),
        mode=args.wandb_mode,
        tags=tags,
        config={
            **config,
            "metadata": {
                "state_indices": metadata["state_indices"],
                "pressure_indices": metadata["pressure_indices"],
                "state_normalization": metadata["state_normalization"],
                "pressure_normalization": metadata["pressure_normalization"],
                "ksteps": metadata["ksteps"],
                "train_drop_last": metadata["train_drop_last"],
                "train_buffer": metadata["train_buffer"],
                "val_buffer": metadata["val_buffer"],
                "no_cross_episode_windows": metadata["no_cross_episode_windows"],
            },
        },
    )
    return run


def average_components(rows: list[dict[str, float]], total_count: int) -> dict[str, float]:
    return {key: sum(row[key] for row in rows) / max(total_count, 1) for key in rows[0]}


@torch.no_grad()
def evaluate(
    model: KoopmanNetwork,
    loader: DataLoader,
    mse_loss: nn.Module,
    device: torch.device,
    u_dim: int,
    gamma: float,
    n_state: int,
    weights: KoopmanLossWeights,
) -> dict[str, float]:
    model.eval()
    weighted_rows: list[dict[str, float]] = []
    total_count = 0
    for (batch,) in loader:
        batch = batch.to(device)
        _, components = define_koopman_loss(batch, model, mse_loss, u_dim, gamma, n_state, weights)
        batch_size = batch.shape[0]
        total_count += batch_size
        weighted_rows.append({key: float(value.item()) * batch_size for key, value in components.items()})
    return average_components(weighted_rows, total_count)


def save_checkpoint(
    path: Path,
    model: KoopmanNetwork,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    config: dict,
    metadata: dict,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "best_val_loss": float(best_val_loss),
            "encode_layers": model.encode_layers,
            "n_koopman": model.n_koopman,
            "u_dim": model.u_dim,
            "config": config,
            "metadata": metadata,
        },
        path,
    )


def train(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    dataset_root = args.dataset_root.resolve()
    full_state_mean, full_state_std, state_names = load_lerobot_state_stats(dataset_root)
    state_indices = parse_int_list(args.state_indices, total_dim=len(full_state_mean))
    pressure_indices = parse_int_list(args.pressure_indices)
    state_mean = full_state_mean[state_indices]
    state_std = np.maximum(full_state_std[state_indices], args.norm_eps)

    pressure_columns, _ = load_pressure_metadata(dataset_root)
    episodes = load_episode_arrays(dataset_root, state_indices, pressure_indices)
    train_episodes, val_episodes = split_episodes(sorted(episodes), args.val_ratio, args.seed)
    train_buffer, train_buffer_stats = build_koopman_buffer(
        episodes,
        train_episodes,
        state_mean,
        state_std,
        args.ksteps,
    )
    val_buffer, val_buffer_stats = build_koopman_buffer(
        episodes,
        val_episodes,
        state_mean,
        state_std,
        args.ksteps,
    )

    if args.max_train_windows > 0 and train_buffer.shape[0] > args.max_train_windows:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(train_buffer.shape[0], size=args.max_train_windows, replace=False)
        train_buffer = train_buffer[keep]
        train_buffer_stats["windows_after_subsample"] = int(train_buffer.shape[0])
    if args.max_val_windows > 0 and val_buffer.shape[0] > args.max_val_windows:
        rng = np.random.default_rng(args.seed + 1)
        keep = rng.choice(val_buffer.shape[0], size=args.max_val_windows, replace=False)
        val_buffer = val_buffer[keep]
        val_buffer_stats["windows_after_subsample"] = int(val_buffer.shape[0])

    device = torch.device(args.device)
    train_loader = DataLoader(
        TensorDataset(torch.as_tensor(train_buffer, dtype=torch.float32)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        drop_last=args.drop_last,
    )
    val_loader = DataLoader(
        TensorDataset(torch.as_tensor(val_buffer, dtype=torch.float32)),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        drop_last=False,
    )

    n_state = len(state_indices)
    u_dim = len(pressure_indices)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    encode_layers = [n_state] + hidden_sizes + [args.encode_dim]
    model = KoopmanNetwork(encode_layers=encode_layers, n_koopman=n_state + args.encode_dim, u_dim=u_dim)
    model.to(device)
    model.float()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse_loss = nn.MSELoss()
    weights = KoopmanLossWeights(
        koopman=args.koopman_lam,
        a_eig=args.a_eig_lam,
        svd=args.svd_lam,
        augment=args.augment_lam,
        pred=args.pred_lam,
    )

    output_dir = make_output_dir(args.output_root.resolve(), args.run_name)
    config = vars(args).copy()
    config["dataset_root"] = str(dataset_root)
    config["output_root"] = str(args.output_root.resolve())
    metadata = {
        "source_reference": str(REPO_ROOT / "motion_control_training/reference/Learning_Koopman_with_Reg_HPN.py"),
        "state_key": "observation.state",
        "state_indices": state_indices,
        "state_names": [state_names[i] for i in state_indices],
        "state_normalization": "lerobot_observation_state_mean_std",
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "pressure_key": "raw_pressure",
        "pressure_indices": pressure_indices,
        "pressure_columns": [pressure_columns[i] for i in pressure_indices],
        "pressure_normalization": "none",
        "ksteps": int(args.ksteps),
        "train_drop_last": bool(args.drop_last),
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
        "train_buffer": train_buffer_stats,
        "val_buffer": val_buffer_stats,
        "buffer_layout": "[sample, Ksteps + 1, raw_pressure_12 + normalized_state_12]",
        "no_cross_episode_windows": True,
    }
    (output_dir / "config.json").write_text(
        json.dumps({"config": config, "metadata": metadata}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    wandb_run = init_wandb(args, output_dir, config, metadata)

    print(f"dataset_root={dataset_root}")
    print(f"device={device} train_windows={train_buffer.shape[0]} val_windows={val_buffer.shape[0]}")
    print(f"encode_layers={encode_layers} n_koopman={n_state + args.encode_dim} ksteps={args.ksteps}")

    metrics_path = output_dir / "metrics.csv"
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    best_val_loss = float("inf")
    best_epoch = 0
    completed_epoch = 0
    epochs_without_improvement = 0
    fieldnames = [
        "epoch",
        "train_loss",
        "train_linear_loss",
        "train_a_eig_loss",
        "train_svd_loss",
        "train_augment_loss",
        "train_pred_loss",
        "val_loss",
        "val_linear_loss",
        "val_a_eig_loss",
        "val_svd_loss",
        "val_augment_loss",
        "val_pred_loss",
        "lr",
        "epoch_seconds",
    ]

    try:
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for epoch in range(1, args.epochs + 1):
                epoch_start = time.perf_counter()
                completed_epoch = epoch
                model.train()
                train_rows: list[dict[str, float]] = []
                train_count = 0
                for (batch,) in train_loader:
                    batch = batch.to(device)
                    loss, components = define_koopman_loss(
                        batch,
                        model,
                        mse_loss,
                        u_dim,
                        args.gamma,
                        n_state,
                        weights,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if args.grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()

                    batch_size = batch.shape[0]
                    train_count += batch_size
                    train_rows.append({key: float(value.item()) * batch_size for key, value in components.items()})

                train_metrics = average_components(train_rows, train_count)
                val_metrics = evaluate(model, val_loader, mse_loss, device, u_dim, args.gamma, n_state, weights)
                row = {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_linear_loss": train_metrics["linear_loss"],
                    "train_a_eig_loss": train_metrics["a_eig_loss"],
                    "train_svd_loss": train_metrics["svd_loss"],
                    "train_augment_loss": train_metrics["augment_loss"],
                    "train_pred_loss": train_metrics["pred_loss"],
                    "val_loss": val_metrics["loss"],
                    "val_linear_loss": val_metrics["linear_loss"],
                    "val_a_eig_loss": val_metrics["a_eig_loss"],
                    "val_svd_loss": val_metrics["svd_loss"],
                    "val_augment_loss": val_metrics["augment_loss"],
                    "val_pred_loss": val_metrics["pred_loss"],
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch_seconds": time.perf_counter() - epoch_start,
                }
                writer.writerow(row)
                f.flush()

                if wandb_run is not None:
                    wandb_run.log(row, step=epoch)

                if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
                    print(
                        f"epoch={epoch:04d} train_loss={row['train_loss']:.8f} "
                        f"train_pred={row['train_pred_loss']:.8f} val_loss={row['val_loss']:.8f} "
                        f"val_pred={row['val_pred_loss']:.8f} epoch_seconds={row['epoch_seconds']:.2f}"
                    )

                if row["val_loss"] < best_val_loss - args.min_delta:
                    best_val_loss = row["val_loss"]
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    save_checkpoint(best_path, model, optimizer, epoch, best_val_loss, config, metadata)
                    if wandb_run is not None:
                        wandb_run.summary["best_epoch"] = best_epoch
                        wandb_run.summary["best_val_loss"] = best_val_loss
                else:
                    epochs_without_improvement += 1

                if args.patience > 0 and epochs_without_improvement >= args.patience:
                    print(
                        f"Early stopping at epoch={epoch}; best_epoch={best_epoch} "
                        f"best_val_loss={best_val_loss:.8f}"
                    )
                    break
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    save_checkpoint(last_path, model, optimizer, completed_epoch, best_val_loss, config, metadata)
    print(f"Saved best checkpoint to {best_path}")
    print(f"Saved last checkpoint to {last_path}")
    return best_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Deep Koopman model from LeRobot episodes and raw pressures.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--state-indices", type=str, default="0:12", help="Use first 12 LeRobot state dims by default.")
    parser.add_argument("--pressure-indices", type=str, default="0:12", help="Use first 12 raw pressure dims by default.")
    parser.add_argument("--ksteps", type=int, default=50)
    parser.add_argument("--encode-dim", type=int, default=12)
    parser.add_argument("--hidden-sizes", type=str, default="64,128,64")
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--drop-last", action="store_true", help="Drop incomplete training batches, matching the reference loop.")
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=80, help="Early-stop epochs without val improvement; <=0 disables.")
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--max-train-windows", type=int, default=0, help="Debug/subsample only; 0 uses all train windows.")
    parser.add_argument("--max-val-windows", type=int, default=0, help="Debug/subsample only; 0 uses all val windows.")
    parser.add_argument("--koopman-lam", type=float, default=10.0)
    parser.add_argument("--a-eig-lam", type=float, default=0.003)
    parser.add_argument("--svd-lam", type=float, default=0.003)
    parser.add_argument("--augment-lam", type=float, default=1.0)
    parser.add_argument("--pred-lam", type=float, default=1.0)
    parser.add_argument("--wandb", action="store_true", help="Log epoch metrics to Weights & Biases.")
    parser.add_argument("--wandb-project", type=str, default="soft-robot-koopman")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-name", type=str, default="")
    parser.add_argument("--wandb-tags", type=str, default="koopman,lerobot,pressure12,state12")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    args.device = resolve_device(args.device)
    train(args)


if __name__ == "__main__":
    main()
