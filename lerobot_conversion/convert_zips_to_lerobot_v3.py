from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from lerobot_conversion_common import (
    UintHistogramStats,
    VectorStats,
    camera_output_name,
    detect_cameras,
    ensure_monotonic_timestamps,
    find_directory_episodes,
    estimate_fps,
    find_episodes,
    image_path,
    infer_state_action,
    load_yaml,
    merge_stats_into_lerobot,
    natural_key,
    quantile_name,
    read_csv_rows,
    read_depth,
    read_rgb,
    row_values,
    safe_directory_names,
    safe_zip_names,
    sha256_file,
    timestamp_values,
    write_json,
)


LOGGER = logging.getLogger("convert_zips_to_lerobot_v3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ZIP robot data to LeRobotDataset v3 with lossless RGB-D sidecar.")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--repo-id")
    parser.add_argument("--config", type=Path, default=Path("conversion_config.yaml"))
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument(
        "--repair-zero-velocity-dropouts",
        action="store_true",
        help="Interpolate state runs whose 6D velocity is all zero, then recompute delta TCP actions.",
    )
    parser.add_argument(
        "--no-repair-zero-velocity-dropouts",
        action="store_true",
        help="Disable zero-velocity state/action repair even if enabled in the config.",
    )
    parser.add_argument(
        "--save-raw-pressure-sidecar",
        action="store_true",
        help="Save aligned raw pressure sidecar arrays: u_p1..u_p12 and u_paw1..u_paw4.",
    )
    parser.add_argument(
        "--no-save-raw-pressure-sidecar",
        action="store_true",
        help="Disable raw pressure sidecar export even if enabled in the config.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def run_text(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, check=False, text=True, capture_output=True).stdout.strip()
    except FileNotFoundError:
        return ""


def check_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.executable,
        "git_rev_parse_head": run_text(["git", "rev-parse", "HEAD"]),
        "ffmpeg_version": run_text(["ffmpeg", "-version"]).splitlines()[:3],
        "ffprobe_version": run_text(["ffprobe", "-version"]).splitlines()[:3],
    }
    try:
        import lerobot

        env["lerobot_file"] = getattr(lerobot, "__file__", None)
        env["pip_show_lerobot"] = run_text([sys.executable, "-m", "pip", "show", "lerobot"])
    except Exception as exc:
        env["lerobot_import_error"] = repr(exc)
    return env


def load_lerobot_dataset_class() -> type[Any]:
    try:
        from lerobot.datasets import LeRobotDataset
    except Exception:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset


def inspect_lerobot_capabilities() -> dict[str, Any]:
    caps: dict[str, Any] = {}
    try:
        cls = load_lerobot_dataset_class()
        caps["LeRobotDataset"] = str(cls)
        for name in ["create", "add_frame", "save_episode", "finalize"]:
            caps[name] = hasattr(cls, name)
        caps["create_signature"] = str(inspect.signature(cls.create)) if hasattr(cls, "create") else None
        caps["add_frame_signature"] = str(inspect.signature(cls.add_frame)) if hasattr(cls, "add_frame") else None
        caps["save_episode_signature"] = str(inspect.signature(cls.save_episode)) if hasattr(cls, "save_episode") else None
        caps["finalize_signature"] = str(inspect.signature(cls.finalize)) if hasattr(cls, "finalize") else None
    except Exception as exc:
        caps["dataset_api_error"] = repr(exc)
    try:
        import lerobot

        root = Path(lerobot.__file__).parent
        text_hits: dict[str, bool] = {}
        for needle in ["DepthEncoderConfig", "depth_output_unit", "is_depth_map", "observation.depth"]:
            text_hits[needle] = any(needle in p.read_text(errors="ignore") for p in root.rglob("*.py"))
        caps.update(text_hits)
    except Exception as exc:
        caps["depth_probe_error"] = repr(exc)
    caps["native_depth_supported"] = bool(caps.get("DepthEncoderConfig") and caps.get("is_depth_map"))
    return caps


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml(args.config)
    if args.input_dir:
        config.setdefault("input", {})["zip_dir"] = str(args.input_dir)
    if args.output_root:
        config.setdefault("output", {})["root"] = str(args.output_root)
    if args.repo_id:
        config.setdefault("output", {})["repo_id"] = args.repo_id
    if args.overwrite:
        config.setdefault("output", {})["overwrite"] = True
    if args.strict:
        config.setdefault("dataset", {})["strict_mode"] = True
    if args.repair_zero_velocity_dropouts:
        config.setdefault("postprocess", {}).setdefault("zero_velocity_dropout_repair", {})["enabled"] = True
    if args.no_repair_zero_velocity_dropouts:
        config.setdefault("postprocess", {}).setdefault("zero_velocity_dropout_repair", {})["enabled"] = False
    if args.save_raw_pressure_sidecar:
        config.setdefault("sidecars", {}).setdefault("raw_pressure", {})["enabled"] = True
    if args.no_save_raw_pressure_sidecar:
        config.setdefault("sidecars", {}).setdefault("raw_pressure", {})["enabled"] = False
    return config


def create_dataset(
    root: Path,
    repo_id: str,
    fps: int,
    robot_type: str,
    features: dict[str, Any],
    use_videos: bool,
    vcodec: str,
) -> Any:
    cls = load_lerobot_dataset_class()
    sig = inspect.signature(cls.create)
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "fps": fps,
        "root": root,
        "robot_type": robot_type,
        "features": features,
        "use_videos": use_videos,
    }
    if "image_writer_processes" in sig.parameters:
        kwargs["image_writer_processes"] = 0
    if "image_writer_threads" in sig.parameters:
        kwargs["image_writer_threads"] = 0
    if "video_backend" in sig.parameters:
        kwargs["video_backend"] = "pyav"
    if "vcodec" in sig.parameters:
        kwargs["vcodec"] = vcodec
    return cls.create(**{k: v for k, v in kwargs.items() if k in sig.parameters})


def feature_spec(shape: tuple[int, ...], dtype: str, names: list[str] | None = None) -> dict[str, Any]:
    spec: dict[str, Any] = {"dtype": dtype, "shape": shape}
    if names:
        spec["names"] = names
    return spec


TCP_POSE_COLUMNS = ["x_pos1", "x_pos2", "x_pos3", "x_ang_radian1", "x_ang_radian2", "x_ang_radian3"]
TCP_STATE_COLUMNS = [
    "x_pos1",
    "x_pos2",
    "x_pos3",
    "x_ang_radian1",
    "x_ang_radian2",
    "x_ang_radian3",
    "x_pos_vel1",
    "x_pos_vel2",
    "x_pos_vel3",
    "x_ang_radian_vel1",
    "x_ang_radian_vel2",
    "x_ang_radian_vel3",
]
PRESSURE_COLUMNS = [f"u_p{i}" for i in range(1, 13)]
RAW_PRESSURE_COLUMNS = [f"u_p{i}" for i in range(1, 13)] + [f"u_paw{i}" for i in range(1, 5)]
GRIPPER_COLUMN = "u_paw2"


def gripper_open_value(raw_value: str | float) -> float:
    value = float(raw_value)
    if value == 0:
        return 1.0
    if value == 3:
        return 0.0
    # Some older logs store interpolated or already-binary values. Keep this
    # deterministic while still making the intended 0/3 mapping explicit.
    return 1.0 if value < 1.5 else 0.0


def _camera_number(camera_dir: str) -> str | None:
    import re

    match = re.search(r"cam(?:era)?_?(\d+)", camera_dir)
    return match.group(1) if match else None


def camera_column(header: list[str], camera_dir: str, kind: str, config: dict[str, Any]) -> str:
    column_map = config.get("cameras", {}).get("columns", {})
    if camera_dir in column_map:
        value = column_map[camera_dir]
        if isinstance(value, dict) and kind in value:
            return str(value[kind])
        if isinstance(value, str):
            return value

    number = _camera_number(camera_dir)
    candidates: list[str] = []
    if kind == "rgb":
        if camera_dir.startswith("images_"):
            candidates.append(camera_dir.replace("images_", "image", 1))
        elif camera_dir.startswith("rgb"):
            candidates.append(camera_dir.replace("rgb", "image", 1))
        if number is not None:
            candidates.extend([f"image{number}", f"image{number}_zed_left", f"rgb{number}"])
    else:
        if camera_dir.startswith("depth_"):
            candidates.append(camera_dir.replace("depth_", "depth", 1))
        if number is not None:
            candidates.append(f"depth{number}")

    for candidate in candidates:
        if candidate in header:
            return candidate
    raise ValueError(f"Could not map {kind} camera directory {camera_dir!r} to a CSV column; tried {candidates}")


def resolve_columns(header: list[str], columns: list[str]) -> list[int]:
    missing = [name for name in columns if name not in header]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return [header.index(name) for name in columns]


def resolve_state_schema(header: list[str], row: list[str], config: dict[str, Any]) -> dict[str, Any]:
    mapping = config.get("mapping", {})
    state_cfg = mapping.get("observation_state", "auto")
    gripper_cfg = mapping.get("gripper", {})
    include_gripper = bool(gripper_cfg.get("include_in_state", state_cfg == "tcp_with_gripper"))

    if state_cfg == "tcp_with_gripper":
        indices = resolve_columns(header, TCP_STATE_COLUMNS)
        names = TCP_STATE_COLUMNS.copy()
    else:
        state_only_mapping = dict(mapping)
        state_only_mapping["action"] = "auto"
        indices, names, _, _ = infer_state_action(header, row, state_only_mapping)

    gripper_index = header.index(GRIPPER_COLUMN) if include_gripper and GRIPPER_COLUMN in header else None
    if include_gripper and gripper_index is None:
        raise ValueError(f"CSV missing gripper column {GRIPPER_COLUMN!r}")
    if gripper_index is not None:
        names = [*names, "gripper_open"]
    return {"indices": indices, "names": names, "gripper_index": gripper_index}


def action_config_mode(config: dict[str, Any]) -> str:
    action_cfg = config.get("mapping", {}).get("action", "auto")
    if isinstance(action_cfg, str):
        return action_cfg
    if isinstance(action_cfg, dict):
        return str(action_cfg.get("type", "columns" if "columns" in action_cfg else "auto"))
    return "columns"


def resolve_action_schema(header: list[str], row: list[str], config: dict[str, Any]) -> dict[str, Any]:
    mapping = config.get("mapping", {})
    action_cfg = mapping.get("action", "auto")
    mode = action_config_mode(config)
    gripper_cfg = mapping.get("gripper", {})
    include_gripper = bool(gripper_cfg.get("include_in_action", mode in {"delta_tcp", "raw_pressure"}))

    if mode == "delta_tcp":
        if isinstance(action_cfg, dict) and isinstance(action_cfg.get("tcp_columns"), list):
            tcp_columns = [str(x) for x in action_cfg["tcp_columns"]]
        else:
            tcp_columns = TCP_POSE_COLUMNS
        indices = resolve_columns(header, tcp_columns)
        names = [f"delta_{name}" for name in tcp_columns]
        gripper_index = header.index(GRIPPER_COLUMN) if include_gripper and GRIPPER_COLUMN in header else None
        if include_gripper and gripper_index is None:
            raise ValueError(f"CSV missing gripper column {GRIPPER_COLUMN!r}")
        if gripper_index is not None:
            names.append("gripper_open")
        return {"mode": mode, "indices": indices, "names": names, "gripper_index": gripper_index}

    if mode == "raw_pressure":
        if isinstance(action_cfg, dict) and isinstance(action_cfg.get("columns"), list):
            pressure_columns = [str(x) for x in action_cfg["columns"]]
        else:
            pressure_columns = PRESSURE_COLUMNS
        indices = resolve_columns(header, pressure_columns)
        names = pressure_columns.copy()
        gripper_index = header.index(GRIPPER_COLUMN) if include_gripper and GRIPPER_COLUMN in header else None
        if include_gripper and gripper_index is None:
            raise ValueError(f"CSV missing gripper column {GRIPPER_COLUMN!r}")
        if gripper_index is not None:
            names.append("gripper_open")
        return {"mode": mode, "indices": indices, "names": names, "gripper_index": gripper_index}

    _, _, indices, names = infer_state_action(header, row, mapping)
    return {"mode": "columns", "indices": indices, "names": names, "gripper_index": None}


def compute_state(row: list[str], schema: dict[str, Any], context: str) -> np.ndarray:
    state = row_values(row, schema["indices"], context)
    if schema.get("gripper_index") is not None:
        state = np.concatenate(
            [state, np.asarray([gripper_open_value(row[int(schema["gripper_index"])])], dtype=np.float32)]
        )
    return state.astype(np.float32, copy=False)


def compute_action(rows: list[list[str]], frame_index: int, schema: dict[str, Any], context: str) -> np.ndarray:
    row = rows[frame_index]
    mode = schema.get("mode")
    if mode == "delta_tcp":
        current = row_values(row, schema["indices"], context)
        if frame_index + 1 < len(rows):
            nxt = row_values(rows[frame_index + 1], schema["indices"], context)
            action = nxt - current
        else:
            action = np.zeros_like(current, dtype=np.float32)
    else:
        action = row_values(row, schema["indices"], context)
    if schema.get("gripper_index") is not None:
        action = np.concatenate(
            [action, np.asarray([gripper_open_value(row[int(schema["gripper_index"])])], dtype=np.float32)]
        )
    return action.astype(np.float32, copy=False)


def raw_pressure_array(header: list[str], rows: list[list[str]]) -> np.ndarray:
    indices = resolve_columns(header, RAW_PRESSURE_COLUMNS)
    values = np.zeros((len(rows), len(indices)), dtype=np.float32)
    for row_i, row in enumerate(rows):
        values[row_i] = row_values(row, indices, f"raw_pressure:frame={row_i}")
    return values


def recompute_delta_tcp_actions_from_state(states: np.ndarray, rows: list[list[str]], action_schema: dict[str, Any]) -> np.ndarray:
    action_dim = len(action_schema["names"])
    actions = np.zeros((len(states), action_dim), dtype=np.float32)
    if len(states) > 1:
        actions[:-1, :6] = states[1:, :6] - states[:-1, :6]
    gripper_index = action_schema.get("gripper_index")
    if gripper_index is not None:
        actions[:, -1] = np.asarray([gripper_open_value(row[int(gripper_index)]) for row in rows], dtype=np.float32)
    return actions


def compute_episode_state_action(
    header: list[str],
    rows: list[list[str]],
    state_schema: dict[str, Any],
    action_schema: dict[str, Any],
    config: dict[str, Any],
    context: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    states = np.stack([compute_state(row, state_schema, f"{context}:frame={i}") for i, row in enumerate(rows)], axis=0)
    actions = np.stack([compute_action(rows, i, action_schema, f"{context}:frame={i}") for i in range(len(rows))], axis=0)
    repair_cfg = config.get("postprocess", {}).get("zero_velocity_dropout_repair", {})
    if not bool(repair_cfg.get("enabled", False)):
        return states, actions, []
    if action_schema.get("mode") != "delta_tcp" or states.shape[1] < 12:
        return states, actions, []

    velocity_atol = float(repair_cfg.get("velocity_atol", 1e-9))
    min_run_length = int(repair_cfg.get("min_run_length", 2))
    repair_reports: list[dict[str, Any]] = []
    vel_zero = np.all(np.isclose(states[:, 6:12], 0.0, atol=velocity_atol), axis=1)
    runs = zero_runs(vel_zero)
    for start, end in runs:
        length = end - start + 1
        if length < min_run_length:
            continue
        prev_idx = start - 1 if start > 0 else None
        next_idx = end + 1 if end + 1 < len(states) else None
        if prev_idx is None and next_idx is None:
            continue
        old_states = states[start : end + 1, :12].copy()
        if prev_idx is not None and next_idx is not None:
            left = states[prev_idx, :12]
            right = states[next_idx, :12]
            for k, idx in enumerate(range(start, end + 1), start=1):
                alpha = k / (length + 1)
                states[idx, :12] = (1.0 - alpha) * left + alpha * right
            method = "linear_between_neighbors"
        elif prev_idx is not None:
            states[start : end + 1, :12] = states[prev_idx, :12]
            method = "copy_previous_neighbor"
        else:
            states[start : end + 1, :12] = states[next_idx, :12]
            method = "copy_next_neighbor"
        repair_reports.append(
            {
                "context": context,
                "frame_start": int(start),
                "frame_end": int(end),
                "length": int(length),
                "method": method,
                "old_max_abs_state_12d": float(np.max(np.abs(old_states))),
                "new_max_abs_state_12d": float(np.max(np.abs(states[start : end + 1, :12]))),
                "old_first_state_12d": old_states[0].astype(float).tolist(),
                "old_last_state_12d": old_states[-1].astype(float).tolist(),
                "new_first_state_12d": states[start, :12].astype(float).tolist(),
                "new_last_state_12d": states[end, :12].astype(float).tolist(),
            }
        )
    if repair_reports and bool(repair_cfg.get("recompute_delta_tcp_action", True)):
        old_actions = actions.copy()
        actions = recompute_delta_tcp_actions_from_state(states, rows, action_schema)
        for report in repair_reports:
            start = int(report["frame_start"])
            end = int(report["frame_end"])
            report["old_max_abs_action_6d"] = float(np.max(np.abs(old_actions[start : end + 1, :6])))
            report["new_max_abs_action_6d"] = float(np.max(np.abs(actions[start : end + 1, :6])))
            report["old_last_action_6d"] = old_actions[end, :6].astype(float).tolist()
            report["new_last_action_6d"] = actions[end, :6].astype(float).tolist()
    return states.astype(np.float32, copy=False), actions.astype(np.float32, copy=False), repair_reports


def zero_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for raw_idx in indices[1:]:
        idx = int(raw_idx)
        if idx == prev + 1:
            prev = idx
        else:
            runs.append((start, prev))
            start = prev = idx
    runs.append((start, prev))
    return runs


def save_episode(dataset: Any, task: str) -> None:
    sig = inspect.signature(dataset.save_episode)
    if "task" in sig.parameters:
        dataset.save_episode(task=task)
    else:
        dataset.save_episode()


def add_frame(dataset: Any, frame: dict[str, Any], task: str) -> None:
    sig = inspect.signature(dataset.add_frame)
    if "task" in sig.parameters and "task" not in frame:
        dataset.add_frame(frame, task=task)
    else:
        frame.setdefault("task", task)
        dataset.add_frame(frame)


def finalize_dataset(dataset: Any) -> None:
    if hasattr(dataset, "finalize"):
        dataset.finalize()


def discover_schema_for_conversion(zip_paths: list[Path], config: dict[str, Any]) -> dict[str, Any]:
    episodes = []
    first: dict[str, Any] | None = None
    for source_path in zip_paths:
        source_episodes = find_episodes(source_path) if source_path.is_file() else find_directory_episodes(source_path)
        for episode in source_episodes:
            episodes.append(episode)
            if first is None:
                if episode.source_type == "zip":
                    source: zipfile.ZipFile | Path
                    zf = zipfile.ZipFile(source_path)
                    source = zf
                else:
                    zf = None
                    source = source_path
                try:
                    names = safe_zip_names(source) if isinstance(source, zipfile.ZipFile) else safe_directory_names(source)
                    header, rows = read_csv_rows(source, episode.csv_path)
                    if not rows:
                        continue
                    rgb_dirs, depth_dirs = detect_cameras(names, episode.episode_dir)
                    state_schema = resolve_state_schema(header, rows[0], config)
                    action_schema = resolve_action_schema(header, rows[0], config)
                    timestamps = timestamp_values(header, rows)
                    rgb_shapes = {}
                    for cam_dir in rgb_dirs:
                        col = camera_column(header, cam_dir, "rgb", config)
                        arr = read_rgb(source, image_path(episode.episode_dir, cam_dir, rows[0][header.index(col)]))
                        rgb_shapes[cam_dir] = tuple(arr.shape)
                    first = {
                        "header": header,
                        "state_schema": state_schema,
                        "state_indices": state_schema["indices"],
                        "state_names": state_schema["names"],
                        "action_schema": action_schema,
                        "action_indices": action_schema["indices"],
                        "action_names": action_schema["names"],
                        "rgb_dirs": rgb_dirs,
                        "depth_dirs": depth_dirs,
                        "rgb_shapes": rgb_shapes,
                        "fps": estimate_fps(timestamps),
                    }
                finally:
                    if zf is not None:
                        zf.close()
    if first is None:
        raise ValueError("No episodes found")
    return {"episodes": episodes, **first}


def filter_episodes(episodes: list[Any], config: dict[str, Any]) -> list[Any]:
    selection = config.get("selection", {})
    keys = selection.get("source_episodes")
    if not keys:
        return episodes
    wanted = {
        (str(item["zip"]), str(item["source_episode_id"]))
        for item in keys
    }
    selected = [
        episode
        for episode in episodes
        if (episode.zip_path.name, episode.source_episode_id) in wanted
    ]
    missing = wanted - {(episode.zip_path.name, episode.source_episode_id) for episode in selected}
    if missing:
        raise ValueError(f"Requested source episodes not found: {sorted(missing)}")
    return selected


def build_features(schema: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    rename = config.get("cameras", {}).get("rename", {})
    features: dict[str, Any] = {
        "observation.state": feature_spec((len(schema["state_names"]),), "float32", schema["state_names"]),
        "action": feature_spec((len(schema["action_names"]),), "float32", schema["action_names"]),
        "source.timestamp": feature_spec((1,), "float64", ["source_timestamp"]),
    }
    for cam_dir, shape in schema["rgb_shapes"].items():
        cam = camera_output_name(cam_dir, rename)
        features[f"observation.images.{cam}"] = feature_spec(tuple(shape), "video", ["height", "width", "channel"])
    return features


def validate_episode_alignment(
    available_names: set[str],
    episode_dir: str,
    header: list[str],
    rows: list[list[str]],
    rgb_dirs: list[str],
    depth_dirs: list[str],
    timestamps: np.ndarray,
    config: dict[str, Any],
) -> None:
    strict = bool(config.get("dataset", {}).get("strict_mode", True))
    ensure_monotonic_timestamps(timestamps, episode_dir)
    for row_i, row in enumerate(rows):
        for cam_dir in rgb_dirs:
            col = camera_column(header, cam_dir, "rgb", config)
            member = image_path(episode_dir, cam_dir, row[header.index(col)])
            if member not in available_names:
                raise ValueError(f"Missing RGB frame {member} at row {row_i}")
        for cam_dir in depth_dirs:
            col = camera_column(header, cam_dir, "depth", config)
            member = image_path(episode_dir, cam_dir, row[header.index(col)])
            if member not in available_names:
                raise ValueError(f"Missing depth frame {member} at row {row_i}")
    if strict and len(rows) == 0:
        raise ValueError(f"{episode_dir}: empty episode")


def make_depth_stats_entry(stats: UintHistogramStats, valid_stats: UintHistogramStats, quantiles: list[float]) -> dict[str, Any]:
    return {
        "all_pixels": stats.as_stats(quantiles),
        "valid_pixels": valid_stats.as_stats(quantiles),
        "quantile_method": "exact_uint_histogram",
    }


def convert(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    input_dir = Path(config["input"].get("zip_dir") or config["input"].get("source_dir") or config["input"].get("input_dir"))
    output_root = Path(config["output"]["root"])
    repo_id = str(config["output"]["repo_id"])
    robot_type = str(config["output"].get("robot_type", "custom_robot"))
    overwrite = bool(config["output"].get("overwrite", False))
    task = str(config["dataset"].get("task_default", "robot manipulation task"))
    use_videos = bool(config["dataset"].get("use_videos", True))
    vcodec = str(config.get("video", {}).get("encoder", "h264"))
    if vcodec == "auto":
        vcodec = "h264"
    quantiles = [float(q) for q in config.get("statistics", {}).get("quantiles", [0.01, 0.10, 0.50, 0.90, 0.99])]
    rename = config.get("cameras", {}).get("rename", {})

    zip_paths = sorted(input_dir.glob("*.zip"), key=lambda p: natural_key(p.name))
    source_paths = zip_paths if zip_paths else [input_dir]
    if not input_dir.exists():
        raise ValueError(f"Input path does not exist: {input_dir}")
    if not zip_paths and not find_directory_episodes(input_dir):
        raise ValueError(f"No zip files or episode directories containing data.csv in {input_dir}")

    env = check_environment()
    caps = inspect_lerobot_capabilities()
    schema = discover_schema_for_conversion(source_paths, config)
    episodes = filter_episodes(schema["episodes"], config)
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    fps = int(config["dataset"].get("fps") if config["dataset"].get("fps") != "auto" else schema["fps"])

    source_hashes = {p.name: sha256_file(p) for p in zip_paths}
    if not source_hashes:
        source_hashes[input_dir.name] = {
            "source_type": "directory",
            "data_csv_count": len(list(input_dir.rglob("data.csv"))),
        }
    features = build_features(schema, config)

    detected_schema = {
        "source_count": len(source_paths),
        "zip_count": len(zip_paths),
        "episode_count_available": len(schema["episodes"]),
        "episode_count_selected": len(episodes),
        "fps": fps,
        "state_dim": len(schema["state_names"]),
        "state_names": schema["state_names"],
        "action_dim": len(schema["action_names"]),
        "action_names": schema["action_names"],
        "rgb_camera_dirs": schema["rgb_dirs"],
        "depth_camera_dirs": schema["depth_dirs"],
        "features": features,
        "lerobot_environment": env,
        "lerobot_capabilities": caps,
    }
    if bool(config.get("output", {}).get("write_detected_schema_to_project", False)):
        write_json(Path(__file__).resolve().parent / "detected_source_schema.json", detected_schema)

    if args.dry_run:
        return {
            "success": True,
            "dry_run": True,
            "source_count": len(source_paths),
            "source_zip_count": len(zip_paths),
            "selected_episode_count": len(episodes),
            "detected_schema": detected_schema,
        }

    if output_root.exists() and not overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {output_root}")

    tmp_base = Path(config["input"].get("temp_dir", str(output_root.parent / ".tmp_lerobot_conversion")))
    tmp_root = tmp_base / f"{output_root.name}.tmp.{os.getpid()}"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.parent.mkdir(parents=True, exist_ok=True)

    errors_path = tmp_root / "conversion_errors.jsonl"
    extra_dir = tmp_root / "meta" / "extra"
    raw_depth_root = tmp_root / "depth_raw"
    save_raw_depth = bool(config.get("cameras", {}).get("depth", {}).get("save_raw_sidecar", True))
    save_raw_pressure = bool(config.get("sidecars", {}).get("raw_pressure", {}).get("enabled", False))
    raw_pressure_root = tmp_root / "raw_pressure"

    report: dict[str, Any] = {
        "success": False,
        "source_count": len(source_paths),
        "source_zip_count": len(zip_paths),
        "converted_episode_count": 0,
        "failed_episode_count": 0,
        "total_frames": 0,
        "fps": fps,
        "state_dim": len(schema["state_names"]),
        "action_dim": len(schema["action_names"]),
        "rgb_cameras": {},
        "depth_cameras": {},
        "quantile_fields_verified": [quantile_name(q) for q in quantiles],
        "video_validation": {},
        "depth_alignment": {},
        "source_hashes": source_hashes,
        "warnings": [],
        "errors": [],
        "lerobot_capabilities": caps,
    }
    if not caps.get("native_depth_supported"):
        if save_raw_depth:
            report["warnings"].append("当前 LeRobot 环境未检测到原生 Depth feature 支持；原始深度仅以无损 sidecar 保存。")
        else:
            report["warnings"].append("当前 LeRobot 环境未检测到原生 Depth feature 支持；已按配置跳过 depth_raw，仅保存 depth video sidecar。")

    dataset = create_dataset(tmp_root, repo_id, fps, robot_type, features, use_videos, vcodec)
    extra_dir.mkdir(parents=True, exist_ok=True)
    if save_raw_depth:
        raw_depth_root.mkdir(parents=True, exist_ok=True)
    if save_raw_pressure:
        raw_pressure_root.mkdir(parents=True, exist_ok=True)
    state_stats = VectorStats(len(schema["state_names"]))
    action_stats = VectorStats(len(schema["action_names"]))
    raw_pressure_stats = VectorStats(len(RAW_PRESSURE_COLUMNS)) if save_raw_pressure else None
    rgb_stats = {camera_output_name(d, rename): UintHistogramStats(channels=3, bins=256) for d in schema["rgb_dirs"]}
    depth_all_stats = {camera_output_name(d, rename): UintHistogramStats(channels=1, bins=65536) for d in schema["depth_dirs"]}
    depth_valid_stats = {camera_output_name(d, rename): UintHistogramStats(channels=1, bins=65536) for d in schema["depth_dirs"]}
    depth_index_rows: list[dict[str, Any]] = []
    raw_pressure_index_rows: list[dict[str, Any]] = []
    raw_pressure_episode_rows: list[dict[str, Any]] = []
    dropout_repair_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    progress_total = sum(1 for _ in episodes)
    pbar = tqdm(total=progress_total, desc="convert episodes")
    for episode_index, episode in enumerate(episodes):
        try:
            if episode.source_type == "zip":
                source_context = zipfile.ZipFile(episode.zip_path)
            else:
                source_context = episode.zip_path
            try:
                source = source_context
                names = safe_zip_names(source) if isinstance(source, zipfile.ZipFile) else safe_directory_names(source)
                available_names = set(names)
                header, rows = read_csv_rows(source, episode.csv_path)
                timestamps = timestamp_values(header, rows)
                rgb_dirs, depth_dirs = detect_cameras(names, episode.episode_dir)
                state_schema = resolve_state_schema(header, rows[0], config)
                action_schema = resolve_action_schema(header, rows[0], config)
                validate_episode_alignment(available_names, episode.episode_dir, header, rows, rgb_dirs, depth_dirs, timestamps, config)
                states, actions, episode_repair_rows = compute_episode_state_action(
                    header,
                    rows,
                    state_schema,
                    action_schema,
                    config,
                    f"{episode.zip_path.name}:{episode.episode_dir}",
                )
                for repair_row in episode_repair_rows:
                    repair_row.update({"episode_index": episode_index, "source_episode_id": episode.source_episode_id})
                dropout_repair_rows.extend(episode_repair_rows)
                raw_pressure_values = raw_pressure_array(header, rows) if save_raw_pressure else None
                if raw_pressure_values is not None:
                    rel_pressure_path = write_raw_pressure_sidecar(tmp_root, episode_index, raw_pressure_values)
                    pressure_sha = sha256_file_like(raw_pressure_values)
                    raw_pressure_episode_rows.append(
                        {
                            "episode_index": episode_index,
                            "frame_count": int(raw_pressure_values.shape[0]),
                            "source_csv": episode.csv_path,
                            "source_episode_id": episode.source_episode_id,
                            "raw_pressure_path": str(rel_pressure_path),
                            "sha256": pressure_sha,
                        }
                    )
                    for frame_index, pressure in enumerate(raw_pressure_values):
                        raw_pressure_stats.update(pressure)
                        raw_pressure_index_rows.append(
                            {
                                "episode_index": episode_index,
                                "frame_index": frame_index,
                                "timestamp": float(timestamps[frame_index]),
                                "raw_pressure_path": str(rel_pressure_path),
                                "raw_array_index": frame_index,
                                "dimension": int(raw_pressure_values.shape[1]),
                                "dtype": str(raw_pressure_values.dtype),
                                "columns": RAW_PRESSURE_COLUMNS,
                                "source_csv": episode.csv_path,
                                "source_episode_id": episode.source_episode_id,
                                "sha256": pressure_sha,
                            }
                        )
                depth_buffers: dict[str, list[np.ndarray]] = {camera_output_name(d, rename): [] for d in depth_dirs}
                depth_sha: dict[tuple[str, int], str] = {}
                for frame_index, row in enumerate(rows):
                    context = f"{episode.zip_path.name}:{episode.episode_dir}:frame={frame_index}"
                    state = states[frame_index]
                    action = actions[frame_index]
                    frame: dict[str, Any] = {
                        "observation.state": state,
                        "action": action,
                        "source.timestamp": np.asarray([float(timestamps[frame_index])], dtype=np.float64),
                        "task": task,
                    }
                    for cam_dir in rgb_dirs:
                        col = camera_column(header, cam_dir, "rgb", config)
                        source_member = image_path(episode.episode_dir, cam_dir, row[header.index(col)])
                        cam = camera_output_name(cam_dir, rename)
                        rgb = read_rgb(source, source_member)
                        frame[f"observation.images.{cam}"] = rgb
                        rgb_stats[cam].update(rgb)
                    for cam_dir in depth_dirs:
                        col = camera_column(header, cam_dir, "depth", config)
                        source_member = image_path(episode.episode_dir, cam_dir, row[header.index(col)])
                        cam = camera_output_name(cam_dir, rename)
                        depth = read_depth(source, source_member)
                        depth_buffers[cam].append(depth)
                        depth_all_stats[cam].update(depth)
                        invalid = config.get("cameras", {}).get("depth", {}).get("invalid_value", 0)
                        valid = depth[(depth != invalid) & np.isfinite(depth)]
                        if valid.size:
                            depth_valid_stats[cam].update(valid)
                        depth_sha[(cam, frame_index)] = sha256_file_like(depth)
                    add_frame(dataset, frame, task)
                    state_stats.update(state)
                    action_stats.update(action)
                    mapping_rows.append(
                        {
                            "episode_index": episode_index,
                            "frame_index": frame_index,
                            "source_zip": episode.zip_path.name,
                            "source_episode_id": episode.source_episode_id,
                            "source_csv": episode.csv_path,
                            "timestamp": float(timestamps[frame_index]),
                        }
                    )
                for cam, depth_list in depth_buffers.items():
                    arr = np.stack(depth_list, axis=0)
                    rel_path = write_depth_array_sidecar(tmp_root, cam, episode_index, arr, config) if save_raw_depth else None
                    video_rel_path = write_depth_preview_video(tmp_root, cam, episode_index, arr, fps, config, report)
                    for frame_index, depth in enumerate(depth_list):
                        depth_index_rows.append(
                            {
                                "episode_index": episode_index,
                                "frame_index": frame_index,
                                "timestamp": float(timestamps[frame_index]),
                                "camera_name": cam,
                                "raw_depth_path": str(rel_path) if rel_path is not None else None,
                                "raw_array_index": frame_index if rel_path is not None else None,
                                "depth_video_path": str(video_rel_path) if video_rel_path is not None else None,
                                "depth_video_frame_index": frame_index if video_rel_path is not None else None,
                                "dtype": str(depth.dtype),
                                "unit": config.get("cameras", {}).get("depth", {}).get("input_unit", "mm"),
                                "height": int(depth.shape[0]),
                                "width": int(depth.shape[1]),
                                "invalid_value": int(config.get("cameras", {}).get("depth", {}).get("invalid_value", 0)),
                                "source_zip": episode.zip_path.name,
                                "source_frame_id": frame_index,
                                "rgb_frame_index": frame_index,
                                "rgb_timestamp": float(timestamps[frame_index]),
                                "depth_timestamp": float(timestamps[frame_index]),
                                "rgb_depth_delta_ms": 0.0,
                                "sha256": depth_sha[(cam, frame_index)],
                            }
                        )
                    report["depth_cameras"].setdefault(cam, {"episode_count": 0, "frame_count": 0, "dtype": str(arr.dtype), "resolution": [int(arr.shape[2]), int(arr.shape[1])]})
                    report["depth_cameras"][cam]["episode_count"] += 1
                    report["depth_cameras"][cam]["frame_count"] += int(arr.shape[0])
                save_episode(dataset, task)
                report["converted_episode_count"] += 1
                report["total_frames"] += len(rows)
            finally:
                if isinstance(source_context, zipfile.ZipFile):
                    source_context.close()
        except Exception as exc:
            report["failed_episode_count"] += 1
            err = {
                "zip": str(episode.zip_path),
                "episode_dir": episode.episode_dir,
                "error": repr(exc),
            }
            report["errors"].append(err)
            errors_path.parent.mkdir(parents=True, exist_ok=True)
            with errors_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(err, ensure_ascii=False) + "\n")
            if not args.continue_on_error:
                raise
        finally:
            pbar.update(1)
    pbar.close()

    finalize_dataset(dataset)
    if bool(config.get("output", {}).get("remove_empty_images_dir", False)):
        images_dir = tmp_root / "images"
        if images_dir.exists() and not any(p.is_file() for p in images_dir.rglob("*")):
            shutil.rmtree(images_dir)
    stats_additions: dict[str, Any] = {
        "observation.state": state_stats.as_stats(quantiles),
        "action": action_stats.as_stats(quantiles),
    }
    for cam, stats in rgb_stats.items():
        stats_additions[f"observation.images.{cam}"] = stats.as_stats(quantiles)
    merge_stats_into_lerobot(tmp_root / "meta" / "stats.json", stats_additions)

    depth_stats = {
        cam: make_depth_stats_entry(depth_all_stats[cam], depth_valid_stats[cam], quantiles)
        for cam in depth_all_stats
        if int(depth_all_stats[cam].count[0]) > 0
    }
    write_json(extra_dir / "depth_stats.json", depth_stats)
    write_depth_index(extra_dir / "depth_index.parquet", depth_index_rows)
    write_json(extra_dir / "camera_calibration.json", build_camera_metadata(schema, config))
    write_json(extra_dir / "depth_metadata.json", build_depth_metadata(schema, config, caps))
    if save_raw_pressure:
        write_raw_pressure_index(extra_dir / "raw_pressure_index.parquet", raw_pressure_index_rows)
        write_json(extra_dir / "raw_pressure_stats.json", {"raw_pressure": raw_pressure_stats.as_stats(quantiles)})
        write_json(
            extra_dir / "raw_pressure_metadata.json",
            build_raw_pressure_metadata(raw_pressure_episode_rows, config, input_dir),
        )
        report["raw_pressure_sidecar"] = {
            "enabled": True,
            "episode_count": len(raw_pressure_episode_rows),
            "frame_count": len(raw_pressure_index_rows),
            "dimension": len(RAW_PRESSURE_COLUMNS),
            "columns": RAW_PRESSURE_COLUMNS,
        }
    else:
        report["raw_pressure_sidecar"] = {"enabled": False}
    write_json(tmp_root / "action_state_dropout_repair_report.json", {"runs": dropout_repair_rows})
    write_json(tmp_root / "source_to_lerobot_mapping.json", {"rows": mapping_rows})
    write_json(tmp_root / "detected_source_schema.json", detected_schema)
    write_json(tmp_root / "conversion_config_resolved.json", config)
    report.update(
        {
            "success": report["failed_episode_count"] == 0,
            "conversion_time": datetime.now(timezone.utc).isoformat(),
            "output_root": str(output_root),
            "repo_id": repo_id,
        }
    )
    for cam, stats in rgb_stats.items():
        source_dir = next((d for d in schema["rgb_dirs"] if camera_output_name(d, rename) == cam), schema["rgb_dirs"][0])
        shape = schema["rgb_shapes"][source_dir]
        pixels_per_frame = int(shape[0] * shape[1])
        report["rgb_cameras"][cam] = {
            "frame_count": int(stats.count[0] // pixels_per_frame) if stats.count[0] else 0,
            "dtype": "uint8",
            "codec": "managed_by_lerobot",
            "fps": fps,
            "decode_ok": None,
        }
    write_json(tmp_root / "conversion_report.json", report)

    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    tmp_root.rename(output_root)
    if not args.keep_temp and tmp_base.exists():
        try:
            tmp_base.rmdir()
        except OSError:
            pass
    return report


def sha256_file_like(arr: np.ndarray) -> str:
    h = __import__("hashlib").sha256()
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def write_depth_array_sidecar(tmp_root: Path, cam: str, episode_index: int, arr: np.ndarray, config: dict[str, Any]) -> Path:
    depth_cfg = config.get("cameras", {}).get("depth", {})
    sidecar_format = str(depth_cfg.get("sidecar_format", "npy")).lower()
    rel_base = Path("depth_raw") / cam / f"episode_{episode_index:06d}"
    (tmp_root / rel_base).parent.mkdir(parents=True, exist_ok=True)
    if sidecar_format == "npz":
        rel_path = rel_base.with_suffix(".npz")
        np.savez_compressed(tmp_root / rel_path, depth=arr)
        return rel_path
    if sidecar_format != "npy":
        raise ValueError(f"Unsupported depth sidecar_format: {sidecar_format!r}")
    rel_path = rel_base.with_suffix(".npy")
    np.save(tmp_root / rel_path, arr)
    return rel_path


def write_depth_preview_video(
    tmp_root: Path,
    cam: str,
    episode_index: int,
    arr: np.ndarray,
    fps: int,
    config: dict[str, Any],
    report: dict[str, Any],
) -> Path | None:
    depth_cfg = config.get("cameras", {}).get("depth", {})
    if not bool(depth_cfg.get("save_video_sidecar", True)):
        return None
    rel_path = Path("depth_videos") / cam / f"episode_{episode_index:06d}.mp4"
    out_path = tmp_root / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    invalid_value = depth_cfg.get("invalid_value", 0)
    valid = arr[(arr != invalid_value) & np.isfinite(arr)]
    if valid.size:
        min_value = float(depth_cfg.get("video_min", np.quantile(valid, 0.01)))
        max_value = float(depth_cfg.get("video_max", np.quantile(valid, 0.99)))
    else:
        min_value, max_value = 0.0, 1.0
    if max_value <= min_value:
        max_value = min_value + 1.0
    preview = np.clip((arr.astype(np.float32) - min_value) / (max_value - min_value), 0.0, 1.0)
    preview = (preview * 255.0).astype(np.uint8)
    height, width = int(preview.shape[1]), int(preview.shape[2])
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, input=np.ascontiguousarray(preview).tobytes(), capture_output=True, check=False)
    except FileNotFoundError:
        report.setdefault("warnings", []).append("ffmpeg not found; skipped depth preview videos")
        return None
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").splitlines()[-3:]
        report.setdefault("warnings", []).append(f"ffmpeg failed for {rel_path}: {' | '.join(stderr)}")
        return None
    return rel_path


def write_depth_index(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def write_raw_pressure_sidecar(tmp_root: Path, episode_index: int, arr: np.ndarray) -> Path:
    rel_path = Path("raw_pressure") / f"episode_{episode_index:06d}.npy"
    out_path = tmp_root / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr.astype(np.float32, copy=False))
    return rel_path


def write_raw_pressure_index(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def build_raw_pressure_metadata(
    episode_rows: list[dict[str, Any]],
    config: dict[str, Any],
    input_dir: Path,
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sidecar_format": "npy",
        "alignment": "episode_index + frame_index",
        "columns": RAW_PRESSURE_COLUMNS,
        "shape_per_frame": [len(RAW_PRESSURE_COLUMNS)],
        "dtype": "float32",
        "episode_count": len(episode_rows),
        "frame_count": int(sum(int(row["frame_count"]) for row in episode_rows)),
        "source_root": str(input_dir),
        "enabled_by_config": config.get("sidecars", {}).get("raw_pressure", {}),
        "episodes": episode_rows,
    }


def build_camera_metadata(schema: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    rename = config.get("cameras", {}).get("rename", {})
    return {
        "camera_name_mapping": {d: camera_output_name(d, rename) for d in schema["rgb_dirs"] + schema["depth_dirs"]},
        "rgb": {camera_output_name(d, rename): {"source_dir": d, "shape": list(schema["rgb_shapes"].get(d, [])), "color_order": "RGB"} for d in schema["rgb_dirs"]},
        "depth": {camera_output_name(d, rename): {"source_dir": d, "aligned_to": "frame_order", "registration": "unknown"} for d in schema["depth_dirs"]},
        "intrinsics": {},
        "distortion_coefficients": {},
        "extrinsics": {},
    }


def build_depth_metadata(schema: dict[str, Any], config: dict[str, Any], caps: dict[str, Any]) -> dict[str, Any]:
    depth_cfg = config.get("cameras", {}).get("depth", {})
    rename = config.get("cameras", {}).get("rename", {})
    return {
        "native_lerobot_depth_requested": depth_cfg.get("native_lerobot_depth", "auto"),
        "native_lerobot_depth_supported": bool(caps.get("native_depth_supported")),
        "sidecar_format": depth_cfg.get("sidecar_format", "npy"),
        "video_sidecar_format": "mp4" if depth_cfg.get("save_video_sidecar", True) else None,
        "depth_unit": depth_cfg.get("input_unit", "mm"),
        "depth_scale": 1.0,
        "invalid_value": depth_cfg.get("invalid_value", 0),
        "depth_min_m": depth_cfg.get("depth_min_m"),
        "depth_max_m": depth_cfg.get("depth_max_m"),
        "timestamp_unit": "seconds",
        "cameras": {camera_output_name(d, rename): {"source_dir": d} for d in schema["depth_dirs"]},
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")
    config = resolve_config(args)
    started = time.time()
    report = convert(config, args)
    LOGGER.info("Finished in %.1fs", time.time() - started)
    LOGGER.info(json.dumps(report, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
