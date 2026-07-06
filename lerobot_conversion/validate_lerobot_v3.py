from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import DataLoader


LOGGER = logging.getLogger("validate_lerobot_v3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a converted LeRobot v3 dataset and RGB-D sidecar.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--check-all-episodes", action="store_true")
    parser.add_argument("--check-depth-alignment", action="store_true")
    parser.add_argument("--check-video-decode", action="store_true")
    parser.add_argument("--check-quantiles", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_lerobot_dataset(root: Path, repo_id: str) -> Any:
    try:
        from lerobot.datasets import LeRobotDataset
    except Exception:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(repo_id=repo_id, root=root, video_backend="pyav")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_stats(root: Path) -> list[str]:
    stats = read_json(root / "meta" / "stats.json")
    required = ["q01", "q10", "q50", "q90", "q99"]
    errors = []
    for key in ["observation.state", "action"]:
        if key not in stats:
            errors.append(f"missing stats key {key}")
            continue
        for q in required:
            if q not in stats[key]:
                errors.append(f"missing {key}.{q}")
        for name, value in stats[key].items():
            arr = np.asarray(value) if isinstance(value, list) else np.asarray([value])
            if not np.isfinite(arr.astype(float)).all():
                errors.append(f"stats {key}.{name} contains NaN/Inf")
    return errors


def video_paths(root: Path) -> list[Path]:
    videos = root / "videos"
    if not videos.exists():
        return []
    return sorted(videos.rglob("*.mp4"))


def validate_video(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(path), "ok": False}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        info.update(
            {
                "codec": stream.codec_context.name,
                "width": stream.codec_context.width,
                "height": stream.codec_context.height,
                "frames_declared": stream.frames,
                "fps": float(stream.average_rate) if stream.average_rate else None,
            }
        )
        decoded = 0
        first_mean = None
        last_mean = None
        for frame in container.decode(video=0):
            arr = frame.to_ndarray(format="rgb24")
            if decoded == 0:
                first_mean = float(arr.mean())
            last_mean = float(arr.mean())
            decoded += 1
        info["frames_decoded"] = decoded
        info["first_frame_mean"] = first_mean
        info["last_frame_mean"] = last_mean
        info["ok"] = decoded > 0 and first_mean is not None and first_mean > 1.0
    return info


def validate_depth_sidecar(root: Path) -> dict[str, Any]:
    index_path = root / "meta" / "extra" / "depth_index.parquet"
    if not index_path.exists():
        return {"ok": False, "error": "missing depth_index.parquet"}
    table = pq.read_table(index_path)
    rows = table.to_pylist()
    if not rows:
        return {"ok": False, "error": "empty depth index"}
    samples = [rows[0], rows[len(rows) // 2], rows[-1]]
    sample_results = []
    for row in samples:
        depth = None
        raw_depth_path = row.get("raw_depth_path")
        if raw_depth_path:
            depth_path = root / raw_depth_path
            if not depth_path.exists():
                return {"ok": False, "error": f"missing sidecar {depth_path}"}
            loaded = np.load(depth_path)
            if isinstance(loaded, np.lib.npyio.NpzFile):
                with loaded as data:
                    depth = data["depth"][int(row["raw_array_index"])]
            else:
                depth = loaded[int(row["raw_array_index"])]
        video_path = row.get("depth_video_path")
        if video_path and not (root / video_path).exists():
            return {"ok": False, "error": f"missing depth video {root / video_path}"}
        if depth is None and not video_path:
            return {"ok": False, "error": "depth row has neither raw_depth_path nor depth_video_path"}
        sample = {
            "camera_name": row["camera_name"],
            "episode_index": int(row["episode_index"]),
            "frame_index": int(row["frame_index"]),
            "raw_depth_path": raw_depth_path,
            "depth_video_path": video_path,
        }
        if depth is not None:
            sample.update(
                {
                    "shape": list(depth.shape),
                    "dtype": str(depth.dtype),
                    "min": float(depth.min()),
                    "max": float(depth.max()),
                }
            )
        sample_results.append(sample)
    return {"ok": True, "row_count": len(rows), "samples": sample_results}


def validate_episode_tables(root: Path) -> dict[str, Any]:
    data_files = sorted((root / "data").rglob("*.parquet"))
    if not data_files:
        return {"ok": False, "error": "no data parquet files"}
    total_rows = 0
    for path in data_files:
        total_rows += pq.read_metadata(path).num_rows
    episode_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    episode_rows = sum(pq.read_metadata(path).num_rows for path in episode_files)
    return {"ok": True, "data_rows": total_rows, "episode_rows": episode_rows}


def validate_dataloader(dataset: Any) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "shape"):
            out[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    return out


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(message)s")
    result: dict[str, Any] = {"root": str(args.root), "repo_id": args.repo_id, "success": False, "errors": []}

    dataset = load_lerobot_dataset(args.root, args.repo_id)
    result["dataset_length"] = len(dataset)
    result["episode_tables"] = validate_episode_tables(args.root)

    if args.check_quantiles:
        result["quantile_errors"] = validate_stats(args.root)
        result["errors"].extend(result["quantile_errors"])

    if args.check_video_decode:
        result["videos"] = [validate_video(path) for path in video_paths(args.root)]
        result["errors"].extend([f"video decode failed: {v['path']}" for v in result["videos"] if not v.get("ok")])

    if args.check_depth_alignment:
        result["depth_sidecar"] = validate_depth_sidecar(args.root)
        if not result["depth_sidecar"].get("ok"):
            result["errors"].append(result["depth_sidecar"].get("error", "depth sidecar failed"))

    result["dataloader_batch"] = validate_dataloader(dataset)
    result["success"] = not result["errors"]
    report_path = args.root / "validation_report.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
