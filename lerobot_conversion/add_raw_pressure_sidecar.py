from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PRESSURE_COLUMNS = [f"u_p{i}" for i in range(1, 13)] + [f"u_paw{i}" for i in range(1, 5)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add aligned raw pressure sidecar to an existing converted dataset.")
    parser.add_argument("--root", type=Path, required=True, help="Converted LeRobot dataset root.")
    parser.add_argument("--source-root", type=Path, required=True, help="Original episode directory root.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def load_source_rows(source_root: Path, source_csv: str) -> list[dict[str, str]]:
    path = source_root / source_csv
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty source CSV: {path}")
    missing = [name for name in PRESSURE_COLUMNS if name not in rows[0]]
    if missing:
        raise ValueError(f"{path} missing pressure columns: {missing}")
    return rows


def vector_stats(values: np.ndarray, quantiles: list[float]) -> dict[str, Any]:
    data = values.astype(np.float64, copy=False)
    stats: dict[str, Any] = {
        "min": data.min(axis=0).tolist(),
        "max": data.max(axis=0).tolist(),
        "mean": data.mean(axis=0).tolist(),
        "std": data.std(axis=0).tolist(),
        "count": [int(data.shape[0])] * data.shape[1],
    }
    for q in quantiles:
        stats[f"q{int(round(q * 100)):02d}"] = np.quantile(data, q, axis=0).tolist()
    return stats


def validate_dataset_rows(root: Path, expected_count: int) -> None:
    data_files = sorted((root / "data").rglob("*.parquet"))
    if not data_files:
        raise ValueError(f"No dataset parquet files under {root / 'data'}")
    total = sum(pq.read_metadata(path).num_rows for path in data_files)
    if total != expected_count:
        raise ValueError(f"Dataset row count {total} != mapping row count {expected_count}")


def main() -> None:
    args = parse_args()
    root = args.root
    source_root = args.source_root
    mapping_path = root / "source_to_lerobot_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(mapping_path)
    mapping_rows = read_json(mapping_path)["rows"]
    validate_dataset_rows(root, len(mapping_rows))

    raw_root = root / "raw_pressure"
    extra_dir = root / "meta" / "extra"
    if raw_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{raw_root} exists; use --overwrite")
        shutil.rmtree(raw_root)
    for stale in [
        extra_dir / "raw_pressure_index.parquet",
        extra_dir / "raw_pressure_metadata.json",
        extra_dir / "raw_pressure_stats.json",
    ]:
        if stale.exists() and not args.overwrite:
            raise FileExistsError(f"{stale} exists; use --overwrite")

    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in mapping_rows:
        by_episode[int(row["episode_index"])].append(row)
    for rows in by_episode.values():
        rows.sort(key=lambda item: int(item["frame_index"]))

    csv_cache: dict[str, list[dict[str, str]]] = {}
    index_rows: list[dict[str, Any]] = []
    all_values: list[np.ndarray] = []
    episode_summaries: list[dict[str, Any]] = []
    raw_root.mkdir(parents=True, exist_ok=True)

    for episode_index in sorted(by_episode):
        rows = by_episode[episode_index]
        arr = np.zeros((len(rows), len(PRESSURE_COLUMNS)), dtype=np.float32)
        source_csvs = {str(row["source_csv"]) for row in rows}
        if len(source_csvs) != 1:
            raise ValueError(f"Episode {episode_index} maps to multiple source CSVs: {sorted(source_csvs)}")
        source_csv = source_csvs.pop()
        if source_csv not in csv_cache:
            csv_cache[source_csv] = load_source_rows(source_root, source_csv)
        source_rows = csv_cache[source_csv]

        for i, row in enumerate(rows):
            frame_index = int(row["frame_index"])
            if i != frame_index:
                raise ValueError(f"Episode {episode_index}: raw array index {i} != frame_index {frame_index}")
            if frame_index >= len(source_rows):
                raise ValueError(f"{source_csv}: missing source row {frame_index}")
            src = source_rows[frame_index]
            arr[i] = np.asarray([float(src[name]) for name in PRESSURE_COLUMNS], dtype=np.float32)

        rel_path = Path("raw_pressure") / f"episode_{episode_index:06d}.npy"
        np.save(root / rel_path, arr)
        digest = sha256_array(arr)
        all_values.append(arr)
        episode_summaries.append(
            {
                "episode_index": episode_index,
                "frame_count": int(arr.shape[0]),
                "source_csv": source_csv,
                "raw_pressure_path": str(rel_path),
                "sha256": digest,
            }
        )

        for i, row in enumerate(rows):
            index_rows.append(
                {
                    "episode_index": episode_index,
                    "frame_index": int(row["frame_index"]),
                    "timestamp": float(row["timestamp"]),
                    "raw_pressure_path": str(rel_path),
                    "raw_array_index": i,
                    "dimension": int(arr.shape[1]),
                    "dtype": str(arr.dtype),
                    "columns": PRESSURE_COLUMNS,
                    "source_csv": source_csv,
                    "source_episode_id": str(row["source_episode_id"]),
                    "sha256": digest,
                }
            )

    values = np.concatenate(all_values, axis=0)
    extra_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(index_rows), extra_dir / "raw_pressure_index.parquet")
    stats = vector_stats(values, [0.01, 0.10, 0.50, 0.90, 0.99])
    write_json(extra_dir / "raw_pressure_stats.json", {"raw_pressure": stats})
    write_json(
        extra_dir / "raw_pressure_metadata.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sidecar_format": "npy",
            "alignment": "episode_index + frame_index",
            "columns": PRESSURE_COLUMNS,
            "shape_per_frame": [len(PRESSURE_COLUMNS)],
            "dtype": "float32",
            "episode_count": len(by_episode),
            "frame_count": int(values.shape[0]),
            "source_root": str(source_root),
            "episodes": episode_summaries,
        },
    )
    print(
        json.dumps(
            {
                "success": True,
                "root": str(root),
                "episode_count": len(by_episode),
                "frame_count": int(values.shape[0]),
                "columns": PRESSURE_COLUMNS,
                "raw_pressure_dir": str(raw_root),
                "index": str(extra_dir / "raw_pressure_index.parquet"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
