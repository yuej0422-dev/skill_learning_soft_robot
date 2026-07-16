from __future__ import annotations

import argparse
import csv
import json
import random
import struct
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motion_control_training.koopman.train_koopman_lerobot import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    load_episode_arrays,
    load_lerobot_state_stats,
    load_pressure_metadata,
    parse_hidden_sizes,
    parse_int_list,
    resolve_device,
    split_episodes,
)

try:
    from .model_fullA_history import (
        FullAHistoryKoopmanNetwork,
        FullAHistoryLossWeights,
        define_fullA_history_loss,
    )
except ImportError:  # pragma: no cover
    from model_fullA_history import (
        FullAHistoryKoopmanNetwork,
        FullAHistoryLossWeights,
        define_fullA_history_loss,
    )


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "runs"
DEFAULT_MAT_DATASET_ROOT = REPO_ROOT / "data_collection/koopman_pressure16"

MI_INT8 = 1
MI_UINT8 = 2
MI_INT16 = 3
MI_UINT16 = 4
MI_INT32 = 5
MI_UINT32 = 6
MI_SINGLE = 7
MI_DOUBLE = 9
MI_INT64 = 12
MI_UINT64 = 13
MI_MATRIX = 14
MI_COMPRESSED = 15
MATLAB_NUMERIC_DTYPES = {
    MI_INT8: "i1",
    MI_UINT8: "u1",
    MI_INT16: "<i2",
    MI_UINT16: "<u2",
    MI_INT32: "<i4",
    MI_UINT32: "<u4",
    MI_SINGLE: "<f4",
    MI_DOUBLE: "<f8",
    MI_INT64: "<i8",
    MI_UINT64: "<u8",
}


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


def matlab_pad8(n_bytes: int) -> int:
    return (int(n_bytes) + 7) & ~7


def read_matlab_tag(buffer: bytes, offset: int) -> tuple[int, int, int, bool]:
    raw_type, raw_size = struct.unpack("<II", buffer[offset : offset + 8])
    small_type = raw_type & 0xFFFF
    small_size = (raw_type >> 16) & 0xFFFF
    if small_size and small_type in MATLAB_NUMERIC_DTYPES:
        return small_type, small_size, offset + 4, True
    return raw_type, raw_size, offset + 8, False


def read_matlab_element(buffer: bytes, offset: int) -> tuple[int, bytes, int]:
    data_type, n_bytes, data_offset, is_small = read_matlab_tag(buffer, offset)
    payload = buffer[data_offset : data_offset + n_bytes]
    if is_small:
        next_offset = offset + 8
    elif data_type == MI_COMPRESSED:
        # scipy.io.savemat writes consecutive miCOMPRESSED blocks without padding.
        next_offset = data_offset + n_bytes
    else:
        next_offset = data_offset + matlab_pad8(n_bytes)
    return data_type, payload, next_offset


def iter_matlab_elements(buffer: bytes, offset: int) -> Sequence[tuple[int, bytes]]:
    while offset + 8 <= len(buffer):
        data_type, payload, offset = read_matlab_element(buffer, offset)
        yield data_type, payload


def parse_matlab_matrix(payload: bytes) -> tuple[str, np.ndarray]:
    offset = 0
    _, _, offset = read_matlab_element(payload, offset)  # array flags
    _, dims_payload, offset = read_matlab_element(payload, offset)
    dims = np.frombuffer(dims_payload, dtype="<i4").astype(int).tolist()
    _, name_payload, offset = read_matlab_element(payload, offset)
    name = name_payload.decode("latin1")
    real_type, real_payload, _ = read_matlab_element(payload, offset)
    dtype = MATLAB_NUMERIC_DTYPES.get(real_type)
    if dtype is None:
        raise ValueError(f"Unsupported MATLAB numeric dtype {real_type} for variable {name!r}")
    array = np.frombuffer(real_payload, dtype=np.dtype(dtype)).copy()
    if dims:
        array = array.reshape(tuple(dims), order="F")
    return name, array


def load_mat_v5_numeric(path: Path) -> dict[str, np.ndarray]:
    buffer = path.read_bytes()
    if len(buffer) < 128 or b"MATLAB 5.0 MAT-file" not in buffer[:128]:
        raise ValueError(f"{path} is not a MATLAB v5 .mat file")

    arrays: dict[str, np.ndarray] = {}
    for data_type, payload in iter_matlab_elements(buffer, 128):
        if data_type == MI_COMPRESSED:
            for sub_type, sub_payload in iter_matlab_elements(zlib.decompress(payload), 0):
                if sub_type == MI_MATRIX:
                    name, array = parse_matlab_matrix(sub_payload)
                    arrays[name] = array
        elif data_type == MI_MATRIX:
            name, array = parse_matlab_matrix(payload)
            arrays[name] = array
    return arrays


def load_manifest(dataset_root: Path) -> dict:
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_dataset_format(dataset_root: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if (dataset_root / "meta/stats.json").exists():
        return "lerobot"
    if list(dataset_root.glob("*.mat")):
        return "mat"
    raise ValueError(f"Could not infer dataset format under {dataset_root}")


def ensure_2d_matrix(array: np.ndarray, key: str, path: Path) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{path.name}:{key} must be 2D, got shape {array.shape}")
    return array.astype(np.float32, copy=False)


def load_mat_episode_arrays(
    dataset_root: Path,
    state_indices: Sequence[int],
    pressure_indices: Sequence[int],
    state_key: str,
    pressure_key: str,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict]:
    mat_files = sorted(dataset_root.glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found under {dataset_root}")

    episodes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    lengths: list[int] = []
    zero_velocity_rows = 0
    zero_velocity_episode_count = 0
    longest_zero_velocity_run = 0
    identical_state_rows = 0
    timing_overrun_sum = 0
    sample_dt_chunks: list[np.ndarray] = []
    first_keys: list[str] | None = None

    for episode_idx, path in enumerate(mat_files):
        arrays = load_mat_v5_numeric(path)
        first_keys = first_keys or sorted(arrays)
        if state_key not in arrays:
            raise KeyError(f"{path.name} missing state key {state_key!r}; keys={sorted(arrays)}")
        if pressure_key not in arrays:
            raise KeyError(f"{path.name} missing pressure key {pressure_key!r}; keys={sorted(arrays)}")

        states_full = ensure_2d_matrix(arrays[state_key], state_key, path)
        pressures_full = ensure_2d_matrix(arrays[pressure_key], pressure_key, path)
        if states_full.shape[0] != pressures_full.shape[0]:
            raise ValueError(
                f"{path.name} state/action row mismatch: {states_full.shape[0]} vs {pressures_full.shape[0]}"
            )
        if max(state_indices) >= states_full.shape[1]:
            raise ValueError(f"{path.name} state_indices exceed {state_key} shape {states_full.shape}")
        if max(pressure_indices) >= pressures_full.shape[1]:
            raise ValueError(f"{path.name} pressure_indices exceed {pressure_key} shape {pressures_full.shape}")

        states = states_full[:, state_indices].astype(np.float32, copy=True)
        pressures = pressures_full[:, pressure_indices].astype(np.float32, copy=True)
        if not np.isfinite(states).all() or not np.isfinite(pressures).all():
            raise ValueError(f"{path.name} contains NaN or Inf in selected state/action arrays")
        if (pressures < -1e-6).any() or (pressures > 1.0 + 1e-6).any():
            raise ValueError(f"{path.name} selected pressures are outside [0, 1]")

        lengths.append(int(states.shape[0]))
        if states.shape[1] >= 12:
            velocity_norm = np.linalg.norm(states[:, 6:12], axis=1)
            zero_mask = velocity_norm < 1e-8
            zero_velocity_rows += int(zero_mask.sum())
            zero_velocity_episode_count += int(zero_mask.any())
            run = 0
            for is_zero in zero_mask:
                run = run + 1 if is_zero else 0
                longest_zero_velocity_run = max(longest_zero_velocity_run, run)
        if states.shape[0] > 1:
            identical_state_rows += int((np.linalg.norm(np.diff(states, axis=0), axis=1) < 1e-9).sum())
        if "sample_time_s" in arrays:
            sample_time = np.asarray(arrays["sample_time_s"], dtype=np.float64).reshape(-1)
            if len(sample_time) > 1:
                sample_dt_chunks.append(np.diff(sample_time))
        if "timing_overrun" in arrays:
            timing_overrun_sum += int(np.asarray(arrays["timing_overrun"]).sum())

        episodes[episode_idx] = (states, pressures)

    total_rows = int(sum(lengths))
    sample_dt = np.concatenate(sample_dt_chunks) if sample_dt_chunks else np.asarray([], dtype=np.float64)
    diagnostics = {
        "mat_file_count": len(mat_files),
        "mat_keys": first_keys or [],
        "episode_len_min": int(min(lengths)),
        "episode_len_max": int(max(lengths)),
        "episode_len_mean": float(np.mean(lengths)),
        "total_rows": total_rows,
        "zero_velocity_rows_lt_1e-8": int(zero_velocity_rows),
        "zero_velocity_episode_count": int(zero_velocity_episode_count),
        "zero_velocity_row_ratio": float(zero_velocity_rows / max(total_rows, 1)),
        "longest_zero_velocity_run_rows": int(longest_zero_velocity_run),
        "identical_consecutive_state_rows_lt_1e-9": int(identical_state_rows),
        "timing_overrun_sum": int(timing_overrun_sum),
    }
    if sample_dt.size:
        diagnostics.update(
            {
                "sample_dt_min": float(sample_dt.min()),
                "sample_dt_p50": float(np.percentile(sample_dt, 50)),
                "sample_dt_p95": float(np.percentile(sample_dt, 95)),
                "sample_dt_max": float(sample_dt.max()),
            }
        )
    return episodes, diagnostics


def compute_state_stats_from_episodes(
    episodes: dict[int, tuple[np.ndarray, np.ndarray]],
    norm_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.concatenate([episode[0] for episode in episodes.values()], axis=0).astype(np.float32)
    return states.mean(axis=0), np.maximum(states.std(axis=0), norm_eps)


def summarize_selected_ranges(episodes: dict[int, tuple[np.ndarray, np.ndarray]]) -> dict[str, list[float]]:
    states = np.concatenate([episode[0] for episode in episodes.values()], axis=0)
    pressures = np.concatenate([episode[1] for episode in episodes.values()], axis=0)
    return {
        "state_min": states.min(axis=0).astype(float).tolist(),
        "state_max": states.max(axis=0).astype(float).tolist(),
        "action_min": pressures.min(axis=0).astype(float).tolist(),
        "action_max": pressures.max(axis=0).astype(float).tolist(),
    }


def context_at(
    states: np.ndarray,
    pressures: np.ndarray,
    t: int,
    history_steps: int,
) -> np.ndarray:
    state_history = states[t - history_steps + 1 : t + 1]
    action_history = pressures[t - history_steps : t]
    if state_history.shape[0] != history_steps or action_history.shape[0] != history_steps:
        raise IndexError(f"Invalid context at t={t} with history_steps={history_steps}")
    return np.concatenate([state_history.reshape(-1), action_history.reshape(-1)], axis=0).astype(np.float32)


def build_window_refs(
    episodes: dict[int, tuple[np.ndarray, np.ndarray]],
    episode_ids: Sequence[int],
    history_steps: int,
    ksteps: int,
    max_windows: int = 0,
    seed: int = 0,
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    skipped_short = 0
    original_frames = 0
    processed_frames = 0
    window_refs: list[tuple[int, int]] = []

    for episode in episode_ids:
        states_raw, _ = episodes[int(episode)]
        original_frames += int(len(states_raw))
        processed_frames += int(len(states_raw))
        if len(states_raw) < history_steps + ksteps + 1:
            skipped_short += 1
            continue
        for t in range(history_steps, len(states_raw) - ksteps):
            window_refs.append((int(episode), int(t)))

    total_candidate_windows = len(window_refs)
    if max_windows > 0 and total_candidate_windows > max_windows:
        rng = np.random.default_rng(seed)
        keep = rng.choice(total_candidate_windows, size=max_windows, replace=False)
        window_refs = [window_refs[int(idx)] for idx in keep]

    if not window_refs:
        raise ValueError(
            "No windows built. Need episode length >= history_steps + ksteps + 1 "
            f"({history_steps + ksteps + 1})."
        )

    stats = {
        "episodes": len(episode_ids),
        "skipped_short_episodes": skipped_short,
        "original_frames": original_frames,
        "processed_frames": processed_frames,
        "windows": int(len(window_refs)),
        "candidate_windows": int(total_candidate_windows),
        "history_steps": int(history_steps),
        "ksteps": int(ksteps),
        "upsample_factor": 1,
    }
    if max_windows > 0 and total_candidate_windows > max_windows:
        stats["windows_after_subsample"] = int(len(window_refs))
    return window_refs, stats


class HistoryKoopmanWindowDataset(Dataset):
    def __init__(
        self,
        episodes: dict[int, tuple[np.ndarray, np.ndarray]],
        episode_ids: Sequence[int],
        state_mean: np.ndarray,
        state_std: np.ndarray,
        history_steps: int,
        ksteps: int,
        max_windows: int = 0,
        seed: int = 0,
        precompute_contexts: bool = True,
    ) -> None:
        self.history_steps = int(history_steps)
        self.ksteps = int(ksteps)
        self.window_refs, self.stats = build_window_refs(
            episodes,
            episode_ids,
            self.history_steps,
            self.ksteps,
            max_windows,
            seed,
        )
        self.precompute_contexts = bool(precompute_contexts)
        self.episode_data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}

        used_episodes = sorted({episode for episode, _ in self.window_refs})
        storage_bytes = 0
        for episode in used_episodes:
            states_raw, pressures_raw = episodes[int(episode)]
            norm_states = ((states_raw - state_mean) / state_std).astype(np.float32)
            pressures = pressures_raw.astype(np.float32, copy=True)
            contexts = self._precompute_contexts(norm_states, pressures) if self.precompute_contexts else None
            storage_bytes += int(norm_states.nbytes + pressures.nbytes + (0 if contexts is None else contexts.nbytes))
            self.episode_data[int(episode)] = (norm_states, pressures, contexts)

        self.stats.update(
            {
                "storage_mode": "lazy_window_dataset",
                "precompute_contexts": self.precompute_contexts,
                "loaded_episode_count": int(len(used_episodes)),
                "dataset_storage_bytes": int(storage_bytes),
                "dataset_storage_mb": float(storage_bytes / (1024 * 1024)),
            }
        )

    def _precompute_contexts(self, states: np.ndarray, pressures: np.ndarray) -> np.ndarray:
        contexts = [
            context_at(states, pressures, t, self.history_steps)
            for t in range(self.history_steps, len(states))
        ]
        return np.stack(contexts, axis=0).astype(np.float32)

    def __len__(self) -> int:
        return len(self.window_refs)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        episode, t = self.window_refs[int(index)]
        states, pressures, contexts = self.episode_data[int(episode)]
        if contexts is None:
            context_sequence = np.stack(
                [context_at(states, pressures, t + offset, self.history_steps) for offset in range(self.ksteps + 1)],
                axis=0,
            )
        else:
            context_start = t - self.history_steps
            context_sequence = contexts[context_start : context_start + self.ksteps + 1]
        return (
            np.ascontiguousarray(context_sequence, dtype=np.float32),
            np.ascontiguousarray(states[t : t + self.ksteps + 1], dtype=np.float32),
            np.ascontiguousarray(pressures[t : t + self.ksteps], dtype=np.float32),
            np.ascontiguousarray(states[t + 1 : t + self.ksteps + 1], dtype=np.float32),
        )


def build_history_koopman_buffer(
    episodes: dict[int, tuple[np.ndarray, np.ndarray]],
    episode_ids: Sequence[int],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    history_steps: int,
    ksteps: int,
    max_windows: int = 0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    contexts: list[np.ndarray] = []
    current_states: list[np.ndarray] = []
    controls: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    skipped_short = 0
    original_frames = 0
    processed_frames = 0
    window_refs: list[tuple[int, int]] = []

    for episode in episode_ids:
        states_raw, pressures = episodes[int(episode)]
        original_frames += int(len(states_raw))
        processed_frames += int(len(states_raw))
        if len(states_raw) < history_steps + ksteps + 1:
            skipped_short += 1
            continue
        for t in range(history_steps, len(states_raw) - ksteps):
            window_refs.append((int(episode), int(t)))

    total_candidate_windows = len(window_refs)
    if max_windows > 0 and total_candidate_windows > max_windows:
        rng = np.random.default_rng(seed)
        keep = rng.choice(total_candidate_windows, size=max_windows, replace=False)
        window_refs = [window_refs[int(idx)] for idx in keep]

    normalized_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for episode, t in window_refs:
        if episode not in normalized_cache:
            states_raw, pressures = episodes[int(episode)]
            norm_states = ((states_raw - state_mean) / state_std).astype(np.float32)
            normalized_cache[episode] = (norm_states, pressures.astype(np.float32))
        norm_states, pressures = normalized_cache[episode]

        context_sequence = np.stack(
            [context_at(norm_states, pressures, t + offset, history_steps) for offset in range(ksteps + 1)],
            axis=0,
        )
        current_state_sequence = norm_states[t : t + ksteps + 1]
        control_sequence = pressures[t : t + ksteps]
        state_target_sequence = norm_states[t + 1 : t + ksteps + 1]
        contexts.append(context_sequence)
        current_states.append(current_state_sequence)
        controls.append(control_sequence)
        targets.append(state_target_sequence)

    if not contexts:
        raise ValueError(
            "No windows built. Need episode length >= history_steps + ksteps + 1 "
            f"({history_steps + ksteps + 1})."
        )

    context_array = np.stack(contexts, axis=0).astype(np.float32)
    state_array = np.stack(current_states, axis=0).astype(np.float32)
    control_array = np.stack(controls, axis=0).astype(np.float32)
    target_array = np.stack(targets, axis=0).astype(np.float32)
    stats = {
        "episodes": len(episode_ids),
        "skipped_short_episodes": skipped_short,
        "original_frames": original_frames,
        "processed_frames": processed_frames,
        "windows": int(context_array.shape[0]),
        "candidate_windows": int(total_candidate_windows),
        "history_steps": int(history_steps),
        "ksteps": int(ksteps),
        "upsample_factor": 1,
    }
    if max_windows > 0 and total_candidate_windows > max_windows:
        stats["windows_after_subsample"] = int(context_array.shape[0])
    return context_array, state_array, control_array, target_array, stats


def tensor_components_to_weighted_float(components: dict[str, torch.Tensor], batch_size: int) -> dict[str, float]:
    return {key: float(value.detach().cpu().item()) * batch_size for key, value in components.items()}


def average_components(rows: list[dict[str, float]], total_count: int) -> dict[str, float]:
    return {key: sum(row[key] for row in rows) / max(total_count, 1) for key in rows[0]}


@torch.no_grad()
def evaluate(
    model: FullAHistoryKoopmanNetwork,
    loader: DataLoader,
    mse_loss: nn.Module,
    device: torch.device,
    gamma: float,
    weights: FullAHistoryLossWeights,
    spectral_radius_limit: float,
    target_std: float,
    svd_min_singular_value: float,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    total_count = 0
    for contexts, states, controls, targets in loader:
        contexts = contexts.to(device)
        states = states.to(device)
        controls = controls.to(device)
        targets = targets.to(device)
        _, components = define_fullA_history_loss(
            contexts,
            states,
            controls,
            targets,
            model,
            mse_loss,
            gamma,
            weights,
            spectral_radius_limit,
            target_std,
            svd_min_singular_value,
        )
        batch_size = contexts.shape[0]
        total_count += batch_size
        rows.append(tensor_components_to_weighted_float(components, batch_size))
    return average_components(rows, total_count)


def save_checkpoint(
    path: Path,
    model: FullAHistoryKoopmanNetwork,
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
            "context_dim": model.context_dim,
            "n_state": model.n_state,
            "u_dim": model.u_dim,
            "encode_dim": model.encode_dim,
            "hidden_sizes": model.hidden_sizes,
            "n_koopman": model.n_koopman,
            "config": config,
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[FullAHistoryKoopmanNetwork, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = FullAHistoryKoopmanNetwork(
        context_dim=int(checkpoint["context_dim"]),
        n_state=int(checkpoint["n_state"]),
        u_dim=int(checkpoint["u_dim"]),
        encode_dim=int(checkpoint["encode_dim"]),
        hidden_sizes=[int(v) for v in checkpoint["hidden_sizes"]],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, checkpoint


def init_wandb(args: argparse.Namespace, output_dir: Path, config: dict, metadata: dict):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("wandb is not installed. Install wandb or omit --wandb.") from exc
    tags = [tag for tag in args.wandb_tags.split(",") if tag.strip()]
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_name or output_dir.name,
        dir=str(output_dir),
        mode=args.wandb_mode,
        tags=tags,
        config={**config, "metadata": metadata},
    )


def gradient_diagnostics(model: FullAHistoryKoopmanNetwork) -> dict[str, float]:
    encoder_grad_sq = 0.0
    for parameter in model.encoder.parameters():
        if parameter.grad is not None:
            encoder_grad_sq += float(parameter.grad.detach().pow(2).sum().cpu())
    return {
        "A_grad_norm": float(model.A.grad.detach().norm().cpu()) if model.A.grad is not None else 0.0,
        "B_grad_norm": float(model.B.grad.detach().norm().cpu()) if model.B.grad is not None else 0.0,
        "bias_grad_norm": float(model.bias.grad.detach().norm().cpu()) if model.bias.grad is not None else 0.0,
        "encoder_grad_norm": float(encoder_grad_sq**0.5),
        "A_latent_to_state_grad_norm": (
            float(model.A.grad[model.n_state :, : model.n_state].detach().norm().cpu())
            if model.A.grad is not None
            else 0.0
        ),
    }


def train(args: argparse.Namespace) -> Path:
    if args.upsample_factor != 1:
        raise ValueError("fullA_history_v2 uses recorded data only. Use --upsample-factor 1.")
    if args.source_hz != args.target_hz:
        raise ValueError("fullA_history_v2 does not resample. Set --source-hz equal to --target-hz.")

    set_seed(args.seed)
    dataset_root = args.dataset_root.resolve()
    dataset_format = infer_dataset_format(dataset_root, args.dataset_format)

    if dataset_format == "lerobot":
        full_state_mean, full_state_std, state_names_full = load_lerobot_state_stats(dataset_root)
        state_indices = parse_int_list(args.state_indices, total_dim=len(full_state_mean))
        pressure_indices = parse_int_list(args.pressure_indices)
        state_mean = full_state_mean[state_indices]
        state_std = np.maximum(full_state_std[state_indices], args.norm_eps)
        pressure_columns_full, _ = load_pressure_metadata(dataset_root)
        episodes = load_episode_arrays(dataset_root, state_indices, pressure_indices)
        state_names = [state_names_full[i] for i in state_indices]
        pressure_columns = [pressure_columns_full[i] for i in pressure_indices]
        dataset_metadata = {
            "dataset_format": "lerobot",
            "state_key": "observation.state",
            "state_normalization": "lerobot_observation_state_mean_std",
            "pressure_key": "raw_pressure",
            "mat_manifest": {},
            "mat_diagnostics": {},
        }
    elif dataset_format == "mat":
        manifest = load_manifest(dataset_root)
        manifest_frequency = manifest.get("frequency_hz")
        if manifest_frequency is not None and abs(float(manifest_frequency) - float(args.source_hz)) > 1e-6:
            raise ValueError(
                f"manifest frequency_hz={manifest_frequency} does not match --source-hz={args.source_hz}"
            )

        first_mat = sorted(dataset_root.glob("*.mat"))[0]
        first_arrays = load_mat_v5_numeric(first_mat)
        if args.mat_state_key not in first_arrays:
            raise KeyError(f"{first_mat.name} missing --mat-state-key {args.mat_state_key!r}")
        if args.mat_pressure_key not in first_arrays:
            raise KeyError(f"{first_mat.name} missing --mat-pressure-key {args.mat_pressure_key!r}")
        state_dim = ensure_2d_matrix(first_arrays[args.mat_state_key], args.mat_state_key, first_mat).shape[1]
        pressure_dim = ensure_2d_matrix(first_arrays[args.mat_pressure_key], args.mat_pressure_key, first_mat).shape[1]
        state_indices = parse_int_list(args.state_indices, total_dim=state_dim)
        pressure_indices = parse_int_list(args.pressure_indices, total_dim=pressure_dim)
        episodes, mat_diagnostics = load_mat_episode_arrays(
            dataset_root,
            state_indices,
            pressure_indices,
            args.mat_state_key,
            args.mat_pressure_key,
        )
        state_mean, state_std = compute_state_stats_from_episodes(episodes, args.norm_eps)
        state_names_full = manifest.get("state_layout") or [f"state_{i}" for i in range(state_dim)]
        state_names = [state_names_full[i] if i < len(state_names_full) else f"state_{i}" for i in state_indices]
        pressure_columns = [f"{args.mat_pressure_key}_{i}" for i in pressure_indices]
        dataset_metadata = {
            "dataset_format": "mat",
            "state_key": args.mat_state_key,
            "state_normalization": "mat_selected_state_mean_std",
            "pressure_key": args.mat_pressure_key,
            "mat_manifest": manifest,
            "mat_diagnostics": {**mat_diagnostics, **summarize_selected_ranges(episodes)},
        }
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_format}")

    n_state = len(state_indices)
    u_dim = len(pressure_indices)
    context_dim = args.history_steps * n_state + args.history_steps * u_dim

    train_episodes, val_episodes = split_episodes(sorted(episodes), args.val_ratio, args.seed)
    train_dataset = HistoryKoopmanWindowDataset(
        episodes,
        train_episodes,
        state_mean,
        state_std,
        args.history_steps,
        args.ksteps,
        args.max_train_windows,
        args.seed,
        precompute_contexts=not args.no_precompute_contexts,
    )
    val_dataset = HistoryKoopmanWindowDataset(
        episodes,
        val_episodes,
        state_mean,
        state_std,
        args.history_steps,
        args.ksteps,
        args.max_val_windows,
        args.seed + 1,
        precompute_contexts=not args.no_precompute_contexts,
    )
    train_stats = train_dataset.stats
    val_stats = val_dataset.stats

    device = torch.device(args.device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        drop_last=args.drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        drop_last=False,
    )

    model = FullAHistoryKoopmanNetwork(
        context_dim=context_dim,
        n_state=n_state,
        u_dim=u_dim,
        encode_dim=args.encode_dim,
        hidden_sizes=parse_hidden_sizes(args.hidden_sizes),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse_loss = nn.MSELoss()
    weights = FullAHistoryLossWeights(
        koopman=args.koopman_lam,
        pred=args.pred_lam,
        stability=args.stability_lam,
        std=args.std_lam,
        identity=args.identity_lam,
        svd=args.svd_lam,
        augment=args.augment_lam,
    )

    output_dir = make_output_dir(args.output_root.resolve(), args.run_name)
    config = vars(args).copy()
    config["dataset_root"] = str(dataset_root)
    config["dataset_format"] = dataset_format
    config["output_root"] = str(args.output_root.resolve())
    metadata = {
        "experiment": "fullA_history_v2",
        "dataset_format": dataset_format,
        "state_key": dataset_metadata["state_key"],
        "state_indices": state_indices,
        "state_names": state_names,
        "state_normalization": dataset_metadata["state_normalization"],
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "pressure_key": dataset_metadata["pressure_key"],
        "pressure_indices": pressure_indices,
        "pressure_columns": pressure_columns,
        "pressure_normalization": "none",
        "pressure_centering": "none",
        "source_hz": float(args.source_hz),
        "target_hz": float(args.target_hz),
        "upsample_factor": int(args.upsample_factor),
        "upsample_method": "none",
        "history_steps": int(args.history_steps),
        "rollout_steps": int(args.ksteps),
        "ksteps": int(args.ksteps),
        "context_dim": int(context_dim),
        "buffer_layout": "context_sequence[K+1], current_state_sequence[K+1], control_sequence[K], state_target_sequence[K]",
        "buffer_mode": "lazy_window_dataset",
        "precompute_contexts": not args.no_precompute_contexts,
        "no_cross_episode_windows": True,
        "train_drop_last": bool(args.drop_last),
        "train_buffer": train_stats,
        "val_buffer": val_stats,
        "mat_manifest": dataset_metadata["mat_manifest"],
        "mat_diagnostics": dataset_metadata["mat_diagnostics"],
    }
    (output_dir / "config.json").write_text(
        json.dumps({"config": config, "metadata": metadata}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    wandb_run = init_wandb(args, output_dir, config, metadata)

    fieldnames = [
        "epoch",
        "train_loss",
        "train_linear_loss",
        "train_pred_loss",
        "train_stability_loss",
        "train_std_loss",
        "train_identity_loss",
        "train_svd_loss",
        "train_augment_loss",
        "val_loss",
        "val_linear_loss",
        "val_pred_loss",
        "val_stability_loss",
        "val_std_loss",
        "val_identity_loss",
        "val_svd_loss",
        "val_augment_loss",
        "train_latent_std_min",
        "train_latent_std_mean",
        "train_latent_std_max",
        "val_latent_std_min",
        "val_latent_std_mean",
        "val_latent_std_max",
        "spectral_radius",
        "eig_abs_mean",
        "eig_abs_max",
        "A_norm",
        "B_norm",
        "A_latent_to_state_norm",
        "A_state_to_latent_norm",
        "A_grad_norm",
        "B_grad_norm",
        "bias_grad_norm",
        "encoder_grad_norm",
        "A_latent_to_state_grad_norm",
        "lr",
        "epoch_seconds",
    ]

    print(f"dataset_root={dataset_root}")
    print(
        f"device={device} train_windows={len(train_dataset)} val_windows={len(val_dataset)} "
        f"history_steps={args.history_steps} ksteps={args.ksteps} context_dim={context_dim}"
    )
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    metrics_path = output_dir / "metrics.csv"
    best_val_loss = float("inf")
    best_epoch = 0
    completed_epoch = 0
    epochs_without_improvement = 0

    try:
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for epoch in range(1, args.epochs + 1):
                start = time.perf_counter()
                completed_epoch = epoch
                model.train()
                train_rows: list[dict[str, float]] = []
                grad_rows: list[dict[str, float]] = []
                train_count = 0
                for contexts, states, controls, targets in train_loader:
                    contexts = contexts.to(device)
                    states = states.to(device)
                    controls = controls.to(device)
                    targets = targets.to(device)
                    loss, components = define_fullA_history_loss(
                        contexts,
                        states,
                        controls,
                        targets,
                        model,
                        mse_loss,
                        args.gamma,
                        weights,
                        args.spectral_radius_limit,
                        args.target_std,
                        args.svd_min_singular_value,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_rows.append(gradient_diagnostics(model))
                    if args.grad_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    batch_size = contexts.shape[0]
                    train_count += batch_size
                    train_rows.append(tensor_components_to_weighted_float(components, batch_size))

                train_metrics = average_components(train_rows, train_count)
                val_metrics = evaluate(
                    model,
                    val_loader,
                    mse_loss,
                    device,
                    args.gamma,
                    weights,
                    args.spectral_radius_limit,
                    args.target_std,
                    args.svd_min_singular_value,
                )
                grad_metrics = {key: float(np.mean([row[key] for row in grad_rows])) for key in grad_rows[0]}
                row = {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_linear_loss": train_metrics["linear_loss"],
                    "train_pred_loss": train_metrics["pred_loss"],
                    "train_stability_loss": train_metrics["stability_loss"],
                    "train_std_loss": train_metrics["std_loss"],
                    "train_identity_loss": train_metrics["identity_loss"],
                    "train_svd_loss": train_metrics["svd_loss"],
                    "train_augment_loss": train_metrics["augment_loss"],
                    "val_loss": val_metrics["loss"],
                    "val_linear_loss": val_metrics["linear_loss"],
                    "val_pred_loss": val_metrics["pred_loss"],
                    "val_stability_loss": val_metrics["stability_loss"],
                    "val_std_loss": val_metrics["std_loss"],
                    "val_identity_loss": val_metrics["identity_loss"],
                    "val_svd_loss": val_metrics["svd_loss"],
                    "val_augment_loss": val_metrics["augment_loss"],
                    "train_latent_std_min": train_metrics["latent_std_min"],
                    "train_latent_std_mean": train_metrics["latent_std_mean"],
                    "train_latent_std_max": train_metrics["latent_std_max"],
                    "val_latent_std_min": val_metrics["latent_std_min"],
                    "val_latent_std_mean": val_metrics["latent_std_mean"],
                    "val_latent_std_max": val_metrics["latent_std_max"],
                    "spectral_radius": val_metrics["spectral_radius"],
                    "eig_abs_mean": val_metrics["eig_abs_mean"],
                    "eig_abs_max": val_metrics["eig_abs_max"],
                    "A_norm": val_metrics["A_norm"],
                    "B_norm": val_metrics["B_norm"],
                    "A_latent_to_state_norm": val_metrics["A_latent_to_state_norm"],
                    "A_state_to_latent_norm": val_metrics["A_state_to_latent_norm"],
                    **grad_metrics,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch_seconds": time.perf_counter() - start,
                }
                writer.writerow(row)
                f.flush()
                if wandb_run is not None:
                    wandb_run.log(row, step=epoch)
                if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
                    print(
                        f"epoch={epoch:04d} train_loss={row['train_loss']:.8f} "
                        f"val_loss={row['val_loss']:.8f} linear={row['val_linear_loss']:.8f} "
                        f"pred={row['val_pred_loss']:.8f} latent_std_mean={row['val_latent_std_mean']:.4f} "
                        f"spectral_radius={row['spectral_radius']:.4f}"
                    )
                if not np.isfinite(row["train_loss"]) or not np.isfinite(row["val_loss"]):
                    raise FloatingPointError(f"NaN/Inf detected at epoch {epoch}: {row}")
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
                    print(f"Early stopping at epoch={epoch}; best_epoch={best_epoch} best_val_loss={best_val_loss:.8f}")
                    break
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    save_checkpoint(last_path, model, optimizer, completed_epoch, best_val_loss, config, metadata)
    print(f"Saved best checkpoint to {best_path}")
    print(f"Saved last checkpoint to {last_path}")
    print(f"best_epoch={best_epoch} best_val_loss={best_val_loss:.8f}")
    return best_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Full-A history-context Koopman v2.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_MAT_DATASET_ROOT)
    parser.add_argument("--dataset-format", type=str, default="auto", choices=["auto", "lerobot", "mat"])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--state-indices", type=str, default="0:12")
    parser.add_argument("--pressure-indices", type=str, default="0:12")
    parser.add_argument("--mat-state-key", type=str, default="X")
    parser.add_argument("--mat-pressure-key", type=str, default="U")
    parser.add_argument("--source-hz", type=float, default=50.0)
    parser.add_argument("--target-hz", type=float, default=50.0)
    parser.add_argument("--upsample-factor", type=int, default=1)
    parser.add_argument("--history-steps", type=int, default=30)
    parser.add_argument("--ksteps", type=int, default=50)
    parser.add_argument("--encode-dim", type=int, default=36)
    parser.add_argument("--hidden-sizes", type=str, default="512,512,256,128")
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-std", type=float, default=1.0)
    parser.add_argument("--koopman-lam", type=float, default=10.0)
    parser.add_argument("--pred-lam", type=float, default=1.0)
    parser.add_argument("--stability-lam", type=float, default=0.01)
    parser.add_argument("--std-lam", type=float, default=0.1)
    parser.add_argument("--identity-lam", type=float, default=1e-4)
    parser.add_argument("--svd-lam", type=float, default=0.0)
    parser.add_argument("--augment-lam", type=float, default=0.0)
    parser.add_argument("--svd-min-singular-value", type=float, default=0.0)
    parser.add_argument("--spectral-radius-limit", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-val-windows", type=int, default=0)
    parser.add_argument("--no-precompute-contexts", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="soft-robot-koopman")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-name", type=str, default="")
    parser.add_argument("--wandb-tags", type=str, default="fullA,history,50hz,koopman,v2,mat")
    parser.add_argument("--wandb-mode", type=str, default="online", choices=["online", "offline", "disabled"])
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    args.device = resolve_device(args.device)
    train(args)


if __name__ == "__main__":
    main()
