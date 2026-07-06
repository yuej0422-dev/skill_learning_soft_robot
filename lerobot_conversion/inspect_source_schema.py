from __future__ import annotations

import argparse
import logging
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from lerobot_conversion_common import (
    camera_output_name,
    detect_cameras,
    estimate_fps,
    find_episodes,
    image_path,
    infer_state_action,
    natural_key,
    read_csv_rows,
    safe_zip_names,
    timestamp_values,
    write_json,
)


LOGGER = logging.getLogger("inspect_source_schema")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect robot ZIP schema before LeRobot conversion.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--max-zips", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "detected_source_schema.json")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def inspect_image(zf: zipfile.ZipFile, member: str) -> dict[str, Any]:
    with zf.open(member) as f:
        im = Image.open(f)
        return {"mode": im.mode, "width": im.size[0], "height": im.size[1], "format": im.format}


def inspect_zip(zip_path: Path) -> dict[str, Any]:
    episodes = find_episodes(zip_path)
    zip_info: dict[str, Any] = {
        "zip_name": zip_path.name,
        "zip_path": str(zip_path),
        "zip_size_bytes": zip_path.stat().st_size,
        "zip_is_episode": False,
        "episode_count": len(episodes),
        "episodes": [],
        "extensions": {},
        "rgb_cameras": {},
        "depth_cameras": {},
    }
    with zipfile.ZipFile(zip_path) as zf:
        names = safe_zip_names(zf)
        zip_info["file_count"] = len(names)
        zip_info["extensions"] = dict(Counter(Path(n).suffix.lower() or "<none>" for n in names))
        for episode in episodes:
            rgb_dirs, depth_dirs = detect_cameras(names, episode.episode_dir)
            header, rows = read_csv_rows(zf, episode.csv_path)
            timestamps = timestamp_values(header, rows)
            if rows:
                state_idx, state_names, action_idx, action_names = infer_state_action(header, rows[0], {"observation_state": "auto", "action": "auto"})
            else:
                state_idx, state_names, action_idx, action_names = [], [], [], []
            ep_info: dict[str, Any] = {
                "source_episode_id": episode.source_episode_id,
                "episode_dir": episode.episode_dir,
                "csv_path": episode.csv_path,
                "frame_count": len(rows),
                "csv_header": header,
                "csv_header_column_count": len(header),
                "first_data_row_column_count": len(rows[0]) if rows else 0,
                "extra_tail_column_count": max((len(rows[0]) if rows else 0) - len(header), 0),
                "timestamp_column": "timestamp" if "timestamp" in header else None,
                "timestamp_first": float(timestamps[0]) if len(timestamps) else None,
                "timestamp_last": float(timestamps[-1]) if len(timestamps) else None,
                "timestamp_monotonic": bool(np.all(np.diff(timestamps) > 0)) if len(timestamps) > 1 else True,
                "fps_estimate": estimate_fps(timestamps),
                "state_indices": state_idx,
                "state_names": state_names,
                "action_indices": action_idx,
                "action_names": action_names,
                "rgb_camera_dirs": rgb_dirs,
                "depth_camera_dirs": depth_dirs,
                "rgb": {},
                "depth": {},
            }
            for cam_dir in rgb_dirs:
                col = "image1" if cam_dir.endswith("cam1") else "image2" if cam_dir.endswith("cam2") else None
                if col and col in header and rows:
                    member = image_path(episode.episode_dir, cam_dir, rows[0][header.index(col)])
                    info = inspect_image(zf, member)
                    info["source_column"] = col
                    info["output_camera_name"] = camera_output_name(cam_dir)
                    info["file_format"] = Path(member).suffix.lower()
                    ep_info["rgb"][cam_dir] = info
                    zip_info["rgb_cameras"].setdefault(cam_dir, info)
            for cam_dir in depth_dirs:
                col = "depth1" if cam_dir.endswith("cam1") else "depth2" if cam_dir.endswith("cam2") else None
                if col and col in header and rows:
                    member = image_path(episode.episode_dir, cam_dir, rows[0][header.index(col)])
                    info = inspect_image(zf, member)
                    info["source_column"] = col
                    info["output_camera_name"] = camera_output_name(cam_dir)
                    info["file_format"] = Path(member).suffix.lower()
                    info["unit"] = "mm"
                    info["invalid_value"] = 0
                    ep_info["depth"][cam_dir] = info
                    zip_info["depth_cameras"].setdefault(cam_dir, info)
            zip_info["episodes"].append(ep_info)
    return zip_info


def summarize_schema(zip_schemas: list[dict[str, Any]]) -> dict[str, Any]:
    total_episodes = sum(z["episode_count"] for z in zip_schemas)
    total_frames = sum(ep["frame_count"] for z in zip_schemas for ep in z["episodes"])
    rgb: dict[str, Any] = {}
    depth: dict[str, Any] = {}
    fps_values: list[float] = []
    state_dims = Counter()
    action_dims = Counter()
    for z in zip_schemas:
        for ep in z["episodes"]:
            fps_values.append(float(ep["fps_estimate"]))
            state_dims[len(ep["state_indices"])] += 1
            action_dims[len(ep["action_indices"])] += 1
            rgb.update(ep["rgb"])
            depth.update(ep["depth"])
    return {
        "version": 1,
        "zip_is_episode": False,
        "source_layout": "zip_contains_multiple_record_directories",
        "zip_count": len(zip_schemas),
        "episode_count": total_episodes,
        "total_frames": total_frames,
        "fps_estimate_median": float(np.median(fps_values)) if fps_values else None,
        "state_dim_histogram": dict(state_dims),
        "action_dim_histogram": dict(action_dims),
        "rgb_cameras": rgb,
        "depth_cameras": depth,
        "zips": zip_schemas,
        "notes": [
            "CSV rows contain named columns plus extra unnamed tail columns.",
            "Auto mapping uses the unnamed tail as observation.state when present.",
            "Depth PNG files are read as lossless uint16 values with unit=mm unless overridden.",
        ],
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")
    zips = sorted(args.input_dir.glob("*.zip"), key=lambda p: natural_key(p.name))
    if args.max_zips:
        zips = zips[: args.max_zips]
    if not zips:
        raise SystemExit(f"No zip files found in {args.input_dir}")
    schemas = []
    for zip_path in zips:
        LOGGER.info("Inspecting %s", zip_path)
        schemas.append(inspect_zip(zip_path))
    output = summarize_schema(schemas)
    write_json(args.output, output)
    LOGGER.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
