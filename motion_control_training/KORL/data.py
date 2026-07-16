from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - depends on runtime env
    raise SystemExit("pyarrow is required to read LeRobot parquet files. Use the soft_vla_cuda conda env.") from exc


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


def parse_float_list(spec: str) -> list[float]:
    return [float(item) for item in spec.split(",") if item.strip()]


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
    files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")
    return files


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def resolve_device(device: str) -> str:
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


def split_episodes(episodes: Sequence[int], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    episode_ids = np.asarray(list(episodes), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(episode_ids)
    val_count = max(1, int(round(len(episode_ids) * val_ratio)))
    val_episodes = sorted(int(v) for v in episode_ids[:val_count])
    train_episodes = sorted(int(v) for v in episode_ids[val_count:])
    return train_episodes, val_episodes


def load_episode_arrays(
    dataset_root: Path,
    state_indices: Sequence[int],
    pressure_indices: Sequence[int],
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict]:
    rows: dict[int, list[tuple[int, np.ndarray]]] = {}
    for path in parquet_files(dataset_root):
        table = pq.read_table(path, columns=["observation.state", "episode_index", "frame_index"])
        for state, episode, frame in zip(
            table["observation.state"].to_pylist(),
            table["episode_index"].to_pylist(),
            table["frame_index"].to_pylist(),
        ):
            rows.setdefault(int(episode), []).append(
                (int(frame), np.asarray(state, dtype=np.float32)[list(state_indices)])
            )

    pressure_columns, episode_to_path = load_pressure_metadata(dataset_root)
    if max(pressure_indices) >= len(pressure_columns):
        raise ValueError(f"Pressure index out of range for {len(pressure_columns)} columns: {pressure_indices}")

    episodes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for episode, frame_states in sorted(rows.items()):
        frame_states.sort(key=lambda item: item[0])
        frames = np.asarray([item[0] for item in frame_states], dtype=np.int64)
        expected = np.arange(len(frames), dtype=np.int64)
        if not np.array_equal(frames, expected):
            raise ValueError(f"Episode {episode} frame_index is not contiguous from 0.")

        raw_pressure = np.load(dataset_root / episode_to_path[int(episode)]).astype(np.float32)
        if frames[-1] >= raw_pressure.shape[0]:
            raise ValueError(
                f"Episode {episode} has frame {frames[-1]} but pressure file has {raw_pressure.shape[0]} frames."
            )
        states = np.stack([item[1] for item in frame_states], axis=0).astype(np.float32)
        pressures = raw_pressure[frames][:, list(pressure_indices)].astype(np.float32)
        episodes[int(episode)] = (states, pressures)

    metadata = {
        "pressure_columns": [pressure_columns[i] for i in pressure_indices],
    }
    return episodes, metadata


def build_transition_dataset(
    episodes: dict[int, tuple[np.ndarray, np.ndarray]],
    episode_ids: Sequence[int],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    target_offset: int,
    reward_scale: float = 1.0,
    reward_state_weights: Sequence[float] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float | int]]:
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    next_observations: list[np.ndarray] = []
    terminals: list[float] = []
    skipped_short = 0

    state_std = np.maximum(state_std.astype(np.float32), 1e-6)
    if reward_state_weights is None:
        reward_weights = np.zeros_like(state_std, dtype=np.float32)
        reward_weights[: min(6, reward_weights.shape[0])] = 1.0
    else:
        reward_weights = np.asarray(reward_state_weights, dtype=np.float32)
        if reward_weights.shape != state_std.shape:
            raise ValueError(
                f"reward_state_weights shape {reward_weights.shape} does not match state shape {state_std.shape}."
            )
    for episode in episode_ids:
        states, pressures = episodes[int(episode)]
        max_start = len(states) - target_offset - 1
        if max_start <= 0:
            skipped_short += 1
            continue
        norm_states = ((states - state_mean) / state_std).astype(np.float32)
        for t in range(max_start):
            target_t = norm_states[t + target_offset]
            target_next = norm_states[t + target_offset + 1]
            observations.append(np.concatenate([norm_states[t], target_t], axis=0))
            actions.append(pressures[t])
            next_observations.append(np.concatenate([norm_states[t + 1], target_next], axis=0))
            state_error = norm_states[t + 1] - target_t
            quadratic_cost = float(np.sum(reward_weights * np.square(state_error)))
            rewards.append(float(-reward_scale * quadratic_cost))
            terminals.append(float(t == max_start - 1))

    if not observations:
        raise ValueError(f"No transitions built. Reduce --target-offset; current value is {target_offset}.")

    dataset = {
        "observations": np.stack(observations, axis=0).astype(np.float32),
        "actions": np.stack(actions, axis=0).astype(np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32).reshape(-1, 1),
        "next_observations": np.stack(next_observations, axis=0).astype(np.float32),
        "terminals": np.asarray(terminals, dtype=np.float32).reshape(-1, 1),
    }
    stats = {
        "episodes": int(len(episode_ids)),
        "skipped_short_episodes": int(skipped_short),
        "transitions": int(dataset["observations"].shape[0]),
        "reward_mean": float(dataset["rewards"].mean()),
        "reward_std": float(dataset["rewards"].std()),
        "reward_type": "-state_error^T Q state_error",
        "reward_state_weights": reward_weights.tolist(),
    }
    return dataset, stats


def normalize_rewards(dataset: dict[str, np.ndarray], eps: float = 1e-6) -> dict[str, float]:
    reward_mean = float(dataset["rewards"].mean())
    reward_std = float(dataset["rewards"].std())
    dataset["rewards"] = (dataset["rewards"] - reward_mean) / max(reward_std, eps)
    return {"reward_mean_before_norm": reward_mean, "reward_std_before_norm": reward_std}


def dataset_action_bounds(dataset: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return dataset["actions"].min(axis=0), dataset["actions"].max(axis=0)


def make_output_dir(output_root: Path, run_name: str | None) -> Path:
    from datetime import datetime

    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def save_json(path: Path, payload: dict) -> None:
    def default(obj: object) -> object:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=default), encoding="utf-8")
