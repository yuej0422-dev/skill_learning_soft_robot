from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import sys as _sys
from pathlib import Path as _Path

_SCRIPTS_DIR = _Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import add_src_to_path

PROJECT_ROOT = add_src_to_path()

from soft_vla.data.dataset_inspector import inspect_dataset, write_reports
from soft_vla.schemas import ACTION_NAMES, STATE_NAMES, validate_action, validate_state


TCP_POSE_COLUMNS = [
    "x_pos1",
    "x_pos2",
    "x_pos3",
    "x_ang_radian1",
    "x_ang_radian2",
    "x_ang_radian3",
]
TCP_VELOCITY_COLUMNS = [
    "x_pos_vel1",
    "x_pos_vel2",
    "x_pos_vel3",
    "x_ang_radian_vel1",
    "x_ang_radian_vel2",
    "x_ang_radian_vel3",
]
CAMERA_COLUMNS = {
    "observation.images.main": ("images_cam1_zed_left", "image1_zed_left"),
    "observation.images.wrist_left": ("images_cam2", "image2"),
    "observation.images.wrist_right": ("images_cam3", "image3"),
}


@dataclass(frozen=True)
class EpisodeRows:
    episode_dir: Path
    rows: list[dict[str, str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="../data_collection/robot_records_6_30_5epis",
        help="Directory containing episode_*/data.csv and RGB image folders.",
    )
    parser.add_argument("--output-dir", default="data/real_robot_records_6_30_5epis_lerobot")
    parser.add_argument("--repo-id", default="local/real_robot_records_6_30_5epis")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Resize RGB images to square size before writing. Use 0 to keep source resolution.",
    )
    parser.add_argument("--task", default="Control the soft robot TCP from three RGB cameras and set the gripper.")
    parser.add_argument("--use-videos", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    candidate = PROJECT_ROOT / p
    if candidate.exists() or str(path).startswith("data/"):
        return candidate
    return (PROJECT_ROOT.parent / p).resolve()


def read_episode_rows(episode_dir: Path) -> EpisodeRows:
    csv_path = episode_dir / "data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 2:
        raise ValueError(f"{episode_dir} must contain at least 2 CSV rows to build delta TCP actions.")
    return EpisodeRows(episode_dir=episode_dir, rows=rows)


def f32_values(row: dict[str, str], columns: list[str], context: str) -> np.ndarray:
    values = []
    for column in columns:
        try:
            values.append(float(row[column]))
        except KeyError as exc:
            raise KeyError(f"{context}: missing CSV column {column}") from exc
        except ValueError as exc:
            raise ValueError(f"{context}: non-numeric CSV value in {column}: {row[column]!r}") from exc
    arr = np.asarray(values, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError(f"{context}: non-finite values in {columns}")
    return arr


def gripper_from_u_paw1(row: dict[str, str], context: str) -> np.float32:
    try:
        raw = float(row["u_paw1"])
    except KeyError as exc:
        raise KeyError(f"{context}: missing CSV column u_paw1") from exc
    except ValueError as exc:
        raise ValueError(f"{context}: non-numeric u_paw1: {row['u_paw1']!r}") from exc
    if abs(raw - 3.0) < 1e-6:
        return np.float32(1.0)
    if abs(raw - 0.0) < 1e-6:
        return np.float32(0.0)
    raise ValueError(f"{context}: u_paw1 must be 0/open or 3/closed, got {raw}")


def load_rgb(path: Path, image_size: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        if image_size > 0 and rgb.size != (image_size, image_size):
            rgb = rgb.resize((image_size, image_size), Image.Resampling.BILINEAR)
        return np.asarray(rgb, dtype=np.uint8)


def image_shapes(first_episode: EpisodeRows, image_size: int) -> dict[str, tuple[int, int, int]]:
    shapes = {}
    first = first_episode.rows[0]
    for key, (folder, column) in CAMERA_COLUMNS.items():
        arr = load_rgb(first_episode.episode_dir / folder / first[column], image_size)
        shapes[key] = tuple(arr.shape)
    return shapes


def build_features(shapes: dict[str, tuple[int, int, int]], use_videos: bool) -> dict:
    image_dtype = "video" if use_videos else "image"
    features = {
        "observation.state": {"dtype": "float32", "shape": (13,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES},
    }
    for key, shape in shapes.items():
        features[key] = {"dtype": image_dtype, "shape": shape, "names": ["height", "width", "channels"]}
    return features


def convert_episode(dataset, episode: EpisodeRows, task: str, episode_index: int, image_size: int) -> dict:
    rows = episode.rows
    report = {
        "episode": episode.episode_dir.name,
        "source_rows": len(rows),
        "converted_frames": len(rows) - 1,
        "first_timestamp": float(rows[0]["timestamp"]),
        "last_used_timestamp": float(rows[-2]["timestamp"]),
        "last_source_timestamp": float(rows[-1]["timestamp"]),
    }
    for frame_index in range(len(rows) - 1):
        row = rows[frame_index]
        next_row = rows[frame_index + 1]
        context = f"{episode.episode_dir.name}:frame={frame_index}"
        tcp_pose = f32_values(row, TCP_POSE_COLUMNS, context)
        tcp_velocity = f32_values(row, TCP_VELOCITY_COLUMNS, context)
        next_tcp_pose = f32_values(next_row, TCP_POSE_COLUMNS, context)
        gripper_state = gripper_from_u_paw1(row, context)
        gripper_action = gripper_from_u_paw1(next_row, f"{episode.episode_dir.name}:frame={frame_index + 1}")
        state = np.concatenate([tcp_pose, tcp_velocity, np.asarray([gripper_state], dtype=np.float32)]).astype(np.float32)
        action = np.concatenate(
            [next_tcp_pose - tcp_pose, np.asarray([gripper_action], dtype=np.float32)]
        ).astype(np.float32)
        validate_state(state)
        validate_action(action)
        frame = {
            "observation.state": state,
            "action": action,
            "task": task,
        }
        for key, (folder, column) in CAMERA_COLUMNS.items():
            frame[key] = load_rgb(episode.episode_dir / folder / row[column], image_size)
        dataset.add_frame(frame)
    dataset.save_episode(parallel_encoding=False)
    return report


def main() -> int:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists. Re-run with --overwrite.")
        shutil.rmtree(output_dir)

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:
        raise RuntimeError("LeRobot is required for conversion.") from exc

    episode_dirs = sorted([p for p in input_dir.glob("episode_*") if p.is_dir()])
    if not episode_dirs:
        raise ValueError(f"No episode_* directories found in {input_dir}")
    episodes = [read_episode_rows(p) for p in episode_dirs]
    shapes = image_shapes(episodes[0], args.image_size)
    features = build_features(shapes, args.use_videos)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=output_dir,
        robot_type="soft_robot_real_records",
        use_videos=args.use_videos,
        image_writer_processes=0,
        image_writer_threads=0,
        batch_encoding_size=1,
    )

    episode_reports = []
    for episode_index, episode in enumerate(episodes):
        episode_reports.append(convert_episode(dataset, episode, args.task, episode_index, args.image_size))
    dataset.finalize()

    conversion_report = {
        "success": True,
        "source_root": str(input_dir),
        "output_root": str(output_dir),
        "repo_id": args.repo_id,
        "fps": args.fps,
        "depth_used": False,
        "image_size": args.image_size if args.image_size > 0 else "source_resolution",
        "pressure_action_used": False,
        "gripper_source": "u_paw1; 0=open->0, 3=closed->1",
        "state": {
            "dim": 13,
            "tcp_pose_columns": TCP_POSE_COLUMNS,
            "tcp_velocity_columns": TCP_VELOCITY_COLUMNS,
            "gripper_index": 12,
        },
        "action": {
            "dim": 7,
            "tcp_delta": "next_tcp_pose - current_tcp_pose for columns x_pos1..x_ang_radian3",
            "gripper_index": 6,
            "gripper_target": "next frame binary u_paw1",
            "last_source_row_dropped_per_episode": True,
        },
        "camera_mapping": {key: {"folder": v[0], "csv_column": v[1], "shape": list(shapes[key])} for key, v in CAMERA_COLUMNS.items()},
        "episodes": episode_reports,
        "total_frames": int(sum(item["converted_frames"] for item in episode_reports)),
    }
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "real_records_conversion.json").write_text(
        json.dumps(conversion_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (reports_dir / "real_records_conversion.md").write_text(
        "# Real Robot Records Conversion\n\n"
        f"- Source: `{input_dir}`\n"
        f"- Output: `{output_dir}`\n"
        f"- Repo ID: `{args.repo_id}`\n"
        f"- Episodes: `{len(episode_reports)}`\n"
        f"- Total frames: `{conversion_report['total_frames']}`\n"
        "- Depth used: `false`\n"
        f"- Image size: `{conversion_report['image_size']}`\n"
        "- Raw pressure action used: `false`\n"
        "- Gripper source: `u_paw1`; `0 -> open/0`, `3 -> closed/1`\n"
        "- Action TCP dimensions: `next_tcp_pose - current_tcp_pose`\n"
        "- Last source row dropped per episode: `true`\n",
        encoding="utf-8",
    )
    result = inspect_dataset(output_dir, repo_id=args.repo_id, expected_episodes=len(episode_reports))
    write_reports(result, reports_dir, name="real_records_dataset_report")
    resolved_config = {
        "dataset": {
            "source": "local",
            "root": str(output_dir.relative_to(PROJECT_ROOT) if output_dir.is_relative_to(PROJECT_ROOT) else output_dir),
            "repo_id": args.repo_id,
            "fps": args.fps,
            "episodes": len(episode_reports),
            "frames": conversion_report["total_frames"],
            "image_keys": list(CAMERA_COLUMNS.keys()),
            "state_key": "observation.state",
            "action_key": "action",
            "task_key": "task",
            "rename_map": {},
        }
    }
    (PROJECT_ROOT / "configs" / "dataset.real_records.resolved.json").write_text(
        json.dumps(resolved_config, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(conversion_report, ensure_ascii=False, indent=2))
    if not result.ok:
        print(json.dumps({"inspection_errors": result.errors}, ensure_ascii=False, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
