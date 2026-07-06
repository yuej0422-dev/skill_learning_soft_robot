from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - exercised only in missing envs
    raise SystemExit("pyarrow is required to read LeRobot parquet files. Install pyarrow first.") from exc

try:
    from .model import FeedforwardPressurePolicy
except ImportError:  # pragma: no cover - direct script execution
    from model import FeedforwardPressurePolicy


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


def parquet_files(dataset_root: Path) -> list[Path]:
    data_dir = dataset_root / "data"
    files = sorted(data_dir.glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")
    return files


def load_training_arrays(
    dataset_root: Path,
    state_indices: Sequence[int],
    pressure_indices: Sequence[int],
    input_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episode_indices: list[np.ndarray] = []
    frame_indices: list[np.ndarray] = []

    for path in parquet_files(dataset_root):
        columns = ["observation.state", "episode_index", "frame_index"]
        if input_mode == "target_state":
            columns.append("action")
        table = pq.read_table(path, columns=columns)
        states.extend(np.asarray(v, dtype=np.float32) for v in table["observation.state"].to_pylist())
        if input_mode == "target_state":
            actions.extend(np.asarray(v, dtype=np.float32) for v in table["action"].to_pylist())
        episode_indices.append(np.asarray(table["episode_index"].to_pylist(), dtype=np.int64))
        frame_indices.append(np.asarray(table["frame_index"].to_pylist(), dtype=np.int64))

    state_array = np.stack(states, axis=0).astype(np.float32)
    if input_mode == "target_state":
        action_array = np.stack(actions, axis=0).astype(np.float32)
        if action_array.shape[1] < max(state_indices) + 1:
            raise ValueError(
                "target_state mode can only use state indices covered by the delta action. "
                f"Got state_indices={list(state_indices)} but action_dim={action_array.shape[1]}."
            )
        state_array[:, state_indices] = state_array[:, state_indices] + action_array[:, state_indices]
    episode_array = np.concatenate(episode_indices, axis=0)
    frame_array = np.concatenate(frame_indices, axis=0)

    pressure_columns, episode_to_path = load_pressure_metadata(dataset_root)
    if max(pressure_indices) >= len(pressure_columns):
        raise ValueError(f"Pressure index out of range for {len(pressure_columns)} columns: {pressure_indices}")

    pressure_array = np.empty((state_array.shape[0], len(pressure_indices)), dtype=np.float32)
    for episode in np.unique(episode_array):
        mask = episode_array == episode
        path = dataset_root / episode_to_path[int(episode)]
        raw_pressure = np.load(path).astype(np.float32)
        frames = frame_array[mask]
        if frames.max(initial=-1) >= raw_pressure.shape[0]:
            raise ValueError(
                f"Episode {episode} has frame index {frames.max()} but pressure file "
                f"contains {raw_pressure.shape[0]} frames."
            )
        pressure_array[mask] = raw_pressure[frames][:, pressure_indices]

    selected_states = state_array[:, state_indices]
    return selected_states, pressure_array, episode_array, frame_array


def split_by_episode(episode_indices: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    episodes = np.unique(episode_indices)
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    val_count = max(1, int(round(len(episodes) * val_ratio)))
    val_episodes = set(int(v) for v in episodes[:val_count])
    val_mask = np.asarray([int(ep) in val_episodes for ep in episode_indices], dtype=bool)
    train_mask = ~val_mask
    return train_mask, val_mask


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_output_dir(output_root: Path, run_name: str | None) -> Path:
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> tuple[float, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    sq_error_sum: np.ndarray | None = None
    with torch.no_grad():
        for state, pressure in loader:
            state = state.to(device)
            pressure = pressure.to(device)
            pred = model(state)
            loss = loss_fn(pred, pressure)
            batch_size = state.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
            sq_error = (pred - pressure).pow(2).sum(dim=0).detach().cpu().numpy()
            sq_error_sum = sq_error if sq_error_sum is None else sq_error_sum + sq_error
    rmse = np.sqrt(sq_error_sum / max(total_count, 1))
    return total_loss / max(total_count, 1), rmse


def save_checkpoint(
    path: Path,
    model: FeedforwardPressurePolicy,
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
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "layer_sizes": model.layer_sizes,
            "state_indices": model.state_indices,
            "state_mean": model.state_mean.detach().cpu(),
            "state_std": model.state_std.detach().cpu(),
            "normalization": "lerobot_observation_state_mean_std",
            "config": config,
            "metadata": metadata,
        },
        path,
    )


def train(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    dataset_root = args.dataset_root.resolve()
    state_mean, state_std, state_names = load_lerobot_state_stats(dataset_root)
    state_indices = parse_int_list(args.state_indices, total_dim=len(state_mean))
    pressure_indices = parse_int_list(args.pressure_indices)
    selected_names = [state_names[i] for i in state_indices]
    pressure_columns, _ = load_pressure_metadata(dataset_root)
    selected_pressure_columns = [pressure_columns[i] for i in pressure_indices]

    raw_states, pressures, episode_indices, _ = load_training_arrays(
        dataset_root=dataset_root,
        state_indices=state_indices,
        pressure_indices=pressure_indices,
        input_mode=args.input_mode,
    )
    train_mask, val_mask = split_by_episode(episode_indices, args.val_ratio, args.seed)

    input_mean = state_mean[state_indices]
    input_std = np.maximum(state_std[state_indices], args.norm_eps)
    normalized_states = (raw_states - input_mean) / input_std

    x_train = torch.as_tensor(normalized_states[train_mask], dtype=torch.float32)
    y_train = torch.as_tensor(pressures[train_mask], dtype=torch.float32)
    x_val = torch.as_tensor(normalized_states[val_mask], dtype=torch.float32)
    y_val = torch.as_tensor(pressures[val_mask], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )

    layer_sizes = [len(state_indices)] + parse_hidden_sizes(args.hidden_sizes) + [len(pressure_indices)]
    model = FeedforwardPressurePolicy(
        layer_sizes=layer_sizes,
        state_mean=np.zeros_like(state_mean),
        state_std=np.ones_like(state_std),
        state_indices=state_indices,
        eps=args.norm_eps,
    )
    device = torch.device(args.device)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    output_dir = make_output_dir(args.output_root.resolve(), args.run_name)
    log_path = output_dir / "metrics.csv"

    config = vars(args).copy()
    config["dataset_root"] = str(dataset_root)
    config["output_root"] = str(args.output_root.resolve())
    metadata = {
        "state_indices": state_indices,
        "state_names": selected_names,
        "pressure_indices": pressure_indices,
        "pressure_columns": selected_pressure_columns,
        "input_mode": args.input_mode,
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "train_episodes": int(len(np.unique(episode_indices[train_mask]))),
        "val_episodes": int(len(np.unique(episode_indices[val_mask]))),
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "input_mean": input_mean.tolist(),
        "input_std": input_std.tolist(),
        "target_normalization": "none",
    }
    (output_dir / "config.json").write_text(
        json.dumps({"config": config, "metadata": metadata}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    best_val_loss = float("inf")
    best_epoch = 0
    completed_epoch = 0
    epochs_without_improvement = 0
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "val_rmse_mean", "lr"],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            completed_epoch = epoch
            model.train()
            train_loss_sum = 0.0
            train_count = 0
            for state, pressure in train_loader:
                state = state.to(device)
                pressure = pressure.to(device)
                pred = model.net(state)
                loss = loss_fn(pred, pressure)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                batch_size = state.shape[0]
                train_loss_sum += float(loss.item()) * batch_size
                train_count += batch_size

            train_loss = train_loss_sum / max(train_count, 1)
            val_loss, val_rmse = evaluate(
                lambda_state_model(model),
                val_loader,
                loss_fn,
                device,
            )
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rmse_mean": float(val_rmse.mean()),
                "lr": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row)
            f.flush()

            if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
                rmse_text = ", ".join(f"{v:.5f}" for v in val_rmse)
                print(
                    f"epoch={epoch:04d} train_loss={train_loss:.8f} "
                    f"val_loss={val_loss:.8f} val_rmse_mean={val_rmse.mean():.5f} "
                    f"val_rmse=[{rmse_text}]"
                )

            if val_loss < best_val_loss - args.min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint(
                    best_path,
                    export_raw_state_model(model, state_mean, state_std),
                    optimizer,
                    epoch,
                    best_val_loss,
                    config,
                    metadata,
                )
            else:
                epochs_without_improvement += 1

            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping at epoch={epoch}; best_epoch={best_epoch} "
                    f"best_val_loss={best_val_loss:.8f}"
                )
                break

    save_checkpoint(
        last_path,
        export_raw_state_model(model, state_mean, state_std),
        optimizer,
        completed_epoch,
        best_val_loss,
        config,
        metadata,
    )
    print(f"Saved best checkpoint to {best_path}")
    print(f"Saved last checkpoint to {last_path}")
    return best_path


class NormalizedStateWrapper(nn.Module):
    def __init__(self, model: FeedforwardPressurePolicy) -> None:
        super().__init__()
        self.model = model

    def forward(self, normalized_state: torch.Tensor) -> torch.Tensor:
        return self.model.net(normalized_state)


def lambda_state_model(model: FeedforwardPressurePolicy) -> nn.Module:
    return NormalizedStateWrapper(model)


def export_raw_state_model(
    trained_normalized_model: FeedforwardPressurePolicy,
    state_mean: np.ndarray,
    state_std: np.ndarray,
) -> FeedforwardPressurePolicy:
    exported = FeedforwardPressurePolicy(
        layer_sizes=trained_normalized_model.layer_sizes,
        state_mean=state_mean,
        state_std=state_std,
        state_indices=trained_normalized_model.state_indices,
        eps=trained_normalized_model.eps,
    )
    exported.net.load_state_dict(trained_normalized_model.net.state_dict())
    exported.to(next(trained_normalized_model.parameters()).device)
    return exported


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train quasi-static state -> feedforward pressure policy.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--state-indices", type=str, default="0:6", help="State dims used as target state; default is TCP pose.")
    parser.add_argument("--pressure-indices", type=str, default="0:12", help="Pressure dims to predict; default is first 12.")
    parser.add_argument(
        "--input-mode",
        type=str,
        default="target_state",
        choices=["target_state", "observation_state"],
        help="target_state uses observation.state + LeRobot delta action for selected dims.",
    )
    parser.add_argument("--hidden-sizes", type=str, default="128,128,64")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=40, help="Early-stop after this many epochs without val improvement; <=0 disables.")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Minimum val-loss improvement required to reset patience.")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    args.device = resolve_device(args.device)
    train(args)


if __name__ == "__main__":
    main()
