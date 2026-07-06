from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from soft_vla.data.feature_mapping import infer_feature_mapping
from soft_vla.data.image_preprocessor import images_are_distinct
from soft_vla.schemas import (
    ACTION_DIM,
    ACTION_NAMES,
    GRIPPER_ACTION_INDEX,
    GRIPPER_STATE_INDEX,
    STATE_DIM,
    STATE_NAMES,
    STATE_UNITS,
    schema_markdown,
    validate_action,
    validate_state,
)


@dataclass
class InspectionResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_lerobot_dataset(root: Path, repo_id: str | None):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:  # pragma: no cover - depends on external package
        raise RuntimeError(f"LeRobot is not importable: {exc}") from exc
    return LeRobotDataset(repo_id=repo_id or "local/synthetic_soft_robot_vla", root=root)


def inspect_dataset(root: str | Path, repo_id: str | None = None, expected_episodes: int | None = None) -> InspectionResult:
    root = Path(root)
    result = InspectionResult(ok=True)
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return InspectionResult(ok=False, errors=[f"Missing {info_path}"])

    info = _read_json(info_path)
    features = info.get("features", {})
    mapping = infer_feature_mapping(features, ["observation.images.main", "observation.images.wrist_left", "observation.images.wrist_right"])
    result.metadata.update(
        {
            "root": str(root),
            "repo_id": repo_id,
            "codebase_version": info.get("codebase_version"),
            "fps": info.get("fps"),
            "total_episodes": info.get("total_episodes"),
            "total_frames": info.get("total_frames"),
            "features": features,
            "image_keys": list(mapping.image_keys),
            "state_key": mapping.state_key,
            "action_key": mapping.action_key,
        }
    )

    if expected_episodes is not None and info.get("total_episodes") != expected_episodes:
        result.errors.append(f"Expected {expected_episodes} episodes, got {info.get('total_episodes')}.")

    state_feature = features.get(mapping.state_key, {})
    action_feature = features.get(mapping.action_key, {})
    if tuple(state_feature.get("shape", ())) != (STATE_DIM,):
        result.errors.append(f"State feature shape must be {(STATE_DIM,)}, got {state_feature.get('shape')}.")
    if state_feature.get("names") != STATE_NAMES:
        result.errors.append(f"State names differ from target schema: {state_feature.get('names')}.")
    if tuple(action_feature.get("shape", ())) != (ACTION_DIM,):
        result.errors.append(f"Action feature shape must be {(ACTION_DIM,)}, got {action_feature.get('shape')}.")
    if action_feature.get("names") != ACTION_NAMES:
        result.errors.append(f"Action names differ from target schema: {action_feature.get('names')}.")
    if len(mapping.image_keys) != 3:
        result.errors.append(f"Expected 3 image keys, got {list(mapping.image_keys)}.")

    try:
        ds = _load_lerobot_dataset(root, repo_id)
        result.metadata["reload_len"] = len(ds)
        if len(ds) > 0:
            sample0 = ds[0]
            state0 = np.asarray(sample0[mapping.state_key])
            action0 = np.asarray(sample0[mapping.action_key])
            validate_state(state0)
            validate_action(action0)
            result.metadata["sample_state_shape"] = list(state0.shape)
            result.metadata["sample_action_shape"] = list(action0.shape)
            imgs = [np.asarray(sample0[k]) for k in mapping.image_keys if k in sample0]
            if len(imgs) == 3:
                result.metadata["sample_image_shapes"] = [list(img.shape) for img in imgs]
                if not images_are_distinct(imgs):
                    result.errors.append("The three camera images are identical for sample 0.")
            task = sample0.get("task")
            if task is not None:
                result.metadata["sample_task"] = str(task)
            elif "task_index" not in sample0:
                result.errors.append("Sample has neither task nor task_index.")
    except Exception as exc:
        result.errors.append(f"Official LeRobotDataset reload/sample failed: {type(exc).__name__}: {exc}")

    try:
        import pandas as pd

        parquet_files = sorted((root / "data").glob("chunk-*/*.parquet"))
        if parquet_files:
            df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
            states = np.stack(df[mapping.state_key].to_numpy())
            actions = np.stack(df[mapping.action_key].to_numpy())
            validate_state(states)
            validate_action(actions)
            result.metadata["parquet_rows"] = int(len(df))
            result.metadata["gripper_state_values"] = sorted(np.unique(states[:, GRIPPER_STATE_INDEX]).astype(int).tolist())
            result.metadata["gripper_action_values"] = sorted(np.unique(actions[:, GRIPPER_ACTION_INDEX]).astype(int).tolist())
            if "timestamp" in df:
                monotonic = True
                for _, ep_df in df.groupby("episode_index"):
                    ts = ep_df["timestamp"].to_numpy()
                    monotonic = monotonic and bool(np.all(np.diff(ts) > -1e-7))
                result.metadata["timestamp_monotonic"] = monotonic
                if not monotonic:
                    result.errors.append("Timestamps are not monotonic inside at least one episode.")
            if "episode_index" in df:
                lengths = df.groupby("episode_index").size().to_dict()
                result.metadata["episode_lengths"] = {str(k): int(v) for k, v in lengths.items()}
        else:
            result.errors.append("No parquet data files found.")
    except Exception as exc:
        result.errors.append(f"Parquet inspection failed: {type(exc).__name__}: {exc}")

    result.ok = len(result.errors) == 0
    return result


def write_reports(result: InspectionResult, reports_dir: str | Path, name: str = "synthetic_dataset_report") -> None:
    out = Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{name}.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"ok": result.ok, "errors": result.errors, "warnings": result.warnings, "metadata": result.metadata},
            f,
            indent=2,
            ensure_ascii=False,
        )
    lines = [
        f"# {name}",
        "",
        f"OK: `{result.ok}`",
        "",
        schema_markdown(),
        "## Dataset Metadata",
        "",
        f"- Root: `{result.metadata.get('root')}`",
        f"- LeRobot codebase version: `{result.metadata.get('codebase_version')}`",
        f"- FPS: `{result.metadata.get('fps')}`",
        f"- Total episodes: `{result.metadata.get('total_episodes')}`",
        f"- Total frames: `{result.metadata.get('total_frames')}`",
        f"- Image keys: `{result.metadata.get('image_keys')}`",
        f"- State key: `{result.metadata.get('state_key')}`",
        f"- Action key: `{result.metadata.get('action_key')}`",
        "",
        "## State Details",
        "",
        f"- Actual state dim: `{result.metadata.get('sample_state_shape')}`",
        f"- State names: `{STATE_NAMES}`",
        f"- State units: `{STATE_UNITS}`",
        "- Missing gripper state: `false`",
        "- Needs user confirmation: `false for current synthetic interface; true before real hardware deployment`",
        "",
        "## Action Details",
        "",
        f"- Actual action dim: `{result.metadata.get('sample_action_shape')}`",
        f"- Rotation representation: `rotation_vector delta, rad`",
        f"- Delta translation unit: `m`",
        f"- Gripper mode: `binary absolute target position`",
        "- Action time alignment: `current observation -> target delta command for next control step`",
        "",
        "## Errors",
        "",
    ]
    lines.extend([f"- {e}" for e in result.errors] or ["- None"])
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {w}" for w in result.warnings] or ["- None"])
    (out / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

