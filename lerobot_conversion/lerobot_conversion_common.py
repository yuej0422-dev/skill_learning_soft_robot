from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image, ImageOps


LOGGER = logging.getLogger("lerobot_conversion")
IGNORED_PARTS = {"__MACOSX", ".DS_Store"}
STATE_HEADER_PREFIX = [
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
ACTION_NAMES = ["u_paw1", "u_paw2", "u_paw3", "u_paw4"]


def natural_key(value: str) -> list[Any]:
    """Return a key where digit runs are sorted as integers."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def safe_zip_names(zf: zipfile.ZipFile, max_member_size: int | None = None) -> list[str]:
    """Return safe regular member names and reject path traversal patterns."""
    names: list[str] = []
    seen: set[str] = set()
    for info in zf.infolist():
        name = info.filename
        if info.is_dir():
            continue
        pure = PurePosixPath(name)
        parts = pure.parts
        if not parts or any(part in IGNORED_PARTS for part in parts):
            continue
        if pure.is_absolute() or any(part == ".." for part in parts):
            raise ValueError(f"Unsafe zip member path: {name}")
        if name in seen:
            raise ValueError(f"Duplicate zip member path: {name}")
        if max_member_size is not None and info.file_size > max_member_size:
            raise ValueError(f"Zip member is too large: {name} ({info.file_size} bytes)")
        seen.add(name)
        names.append(name)
    return names


@dataclass(frozen=True)
class EpisodeRef:
    zip_path: Path
    episode_dir: str
    csv_path: str
    source_episode_id: str
    source_type: str = "zip"


def episode_id_from_dir(episode_dir: str) -> str:
    parts = PurePosixPath(episode_dir).parts
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return episode_dir


def find_episodes(zip_path: Path) -> list[EpisodeRef]:
    with zipfile.ZipFile(zip_path) as zf:
        names = safe_zip_names(zf)
    csvs = sorted([n for n in names if n.endswith("/data.csv")], key=natural_key)
    episodes: list[EpisodeRef] = []
    for csv_path in csvs:
        episode_dir = str(PurePosixPath(csv_path).parent)
        episodes.append(
            EpisodeRef(
                zip_path=zip_path,
                episode_dir=episode_dir,
                csv_path=csv_path,
                source_episode_id=episode_id_from_dir(episode_dir),
            )
        )
    return episodes


def safe_directory_names(root: Path) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        parts = PurePosixPath(rel).parts
        if not parts or any(part in IGNORED_PARTS for part in parts):
            continue
        if any(part == ".." for part in parts):
            raise ValueError(f"Unsafe directory member path: {rel}")
        if rel in seen:
            raise ValueError(f"Duplicate directory member path: {rel}")
        seen.add(rel)
        names.append(rel)
    return names


def find_directory_episodes(root: Path) -> list[EpisodeRef]:
    csvs = sorted([p for p in root.rglob("data.csv") if p.is_file()], key=lambda p: natural_key(p.relative_to(root).as_posix()))
    episodes: list[EpisodeRef] = []
    for csv_file in csvs:
        csv_path = csv_file.relative_to(root).as_posix()
        episode_dir = str(PurePosixPath(csv_path).parent)
        episodes.append(
            EpisodeRef(
                zip_path=root,
                episode_dir=episode_dir,
                csv_path=csv_path,
                source_episode_id=episode_id_from_dir(episode_dir),
                source_type="directory",
            )
        )
    return episodes


def read_member(source: zipfile.ZipFile | Path, path: str) -> bytes:
    if isinstance(source, Path):
        return (source / path).read_bytes()
    return source.read(path)


def read_csv_rows(source: zipfile.ZipFile | Path, csv_path: str) -> tuple[list[str], list[list[str]]]:
    raw = read_member(source, csv_path).decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(raw))
    rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")
    return rows[0], rows[1:]


def numeric_row(row: list[str], row_context: str) -> list[float]:
    out: list[float] = []
    for i, value in enumerate(row):
        if i in (1, 2, 3, 4):
            continue
        try:
            out.append(float(value))
        except ValueError as exc:
            raise ValueError(f"Non-numeric field at {row_context}, column {i}: {value!r}") from exc
    return out


def _resolve_column_indices(spec: Any, header: list[str]) -> tuple[list[int], list[str]]:
    if isinstance(spec, list):
        indices = [header.index(item) if isinstance(item, str) else int(item) for item in spec]
        names = [header[i] if 0 <= i < len(header) else f"column_{i}" for i in indices]
        return indices, names
    if isinstance(spec, dict):
        columns = spec.get("columns")
        if not isinstance(columns, list):
            raise ValueError(f"Column mapping requires a columns list: {spec!r}")
        indices = [header.index(item) if isinstance(item, str) else int(item) for item in columns]
        raw_names = spec.get("names")
        if raw_names is None:
            names = [header[i] if 0 <= i < len(header) else f"column_{i}" for i in indices]
        else:
            if not isinstance(raw_names, list) or len(raw_names) != len(indices):
                raise ValueError("Mapping names must be a list with the same length as columns")
            names = [str(name) for name in raw_names]
        return indices, names
    raise ValueError(f"Unsupported column mapping: {spec!r}")


def infer_state_action(header: list[str], row: list[str], mapping: dict[str, Any]) -> tuple[list[int], list[str], list[int], list[str]]:
    """Infer state/action source indices from a CSV row.

    The current soft-arm logs have named columns up to u_paw4 and then append a
    second 12D TCP state without extending the header. The default auto rule
    preserves that tail as observation.state because it is the supervised target
    used by the previous pipeline.
    """
    state_cfg = mapping.get("observation_state", "auto")
    action_cfg = mapping.get("action", "auto")
    if state_cfg == "auto":
        if len(row) > len(header):
            state_indices = list(range(len(header), len(row)))
            state_names = [f"tcp_state_{i}" for i in range(len(state_indices))]
        else:
            state_indices = [header.index(name) for name in STATE_HEADER_PREFIX if name in header]
            state_names = [header[i] for i in state_indices]
    elif isinstance(state_cfg, (list, dict)):
        state_indices, state_names = _resolve_column_indices(state_cfg, header)
    else:
        raise ValueError(f"Unsupported mapping.observation_state: {state_cfg!r}")

    if action_cfg == "auto":
        action_indices = [header.index(name) for name in ACTION_NAMES if name in header]
        action_names = [header[i] for i in action_indices]
    elif isinstance(action_cfg, (list, dict)):
        action_indices, action_names = _resolve_column_indices(action_cfg, header)
    else:
        raise ValueError(f"Unsupported mapping.action: {action_cfg!r}")

    if not state_indices:
        raise ValueError("Could not infer observation.state columns")
    if not action_indices:
        raise ValueError("Could not infer action columns")
    return state_indices, state_names, action_indices, action_names


def row_values(row: list[str], indices: list[int], context: str) -> np.ndarray:
    values: list[float] = []
    for idx in indices:
        if idx >= len(row):
            raise ValueError(f"{context}: missing CSV column {idx}; row has {len(row)} fields")
        try:
            values.append(float(row[idx]))
        except ValueError as exc:
            raise ValueError(f"{context}: non-numeric CSV value at column {idx}: {row[idx]!r}") from exc
    arr = np.asarray(values, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError(f"{context}: contains NaN/Inf")
    return arr


def timestamp_values(header: list[str], rows: list[list[str]]) -> np.ndarray:
    if "timestamp" not in header:
        return np.arange(len(rows), dtype=np.float64)
    idx = header.index("timestamp")
    stamps = np.asarray([float(row[idx]) for row in rows], dtype=np.float64)
    return stamps


def estimate_fps(timestamps: np.ndarray) -> float:
    if len(timestamps) < 2:
        return 1.0
    diffs = np.diff(timestamps.astype(np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return 1.0
    return float(round(1.0 / float(np.median(diffs))))


def ensure_monotonic_timestamps(timestamps: np.ndarray, context: str) -> None:
    if len(timestamps) <= 1:
        return
    diffs = np.diff(timestamps)
    if not np.all(diffs > 0):
        bad = int(np.where(diffs <= 0)[0][0])
        raise ValueError(f"{context}: timestamp is not strictly increasing at frame {bad}->{bad + 1}")


def detect_cameras(names: Iterable[str], episode_dir: str) -> tuple[list[str], list[str]]:
    prefix = episode_dir.rstrip("/") + "/"
    subdirs: set[str] = set()
    for name in names:
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix) :]
        parts = rest.split("/")
        if len(parts) >= 2:
            subdirs.add(parts[0])
    rgb = sorted([s for s in subdirs if s.startswith("images_") or s.startswith("rgb")], key=natural_key)
    depth = sorted([s for s in subdirs if s.startswith("depth_")], key=natural_key)
    return rgb, depth


def camera_output_name(source_name: str, rename: dict[str, str] | None = None) -> str:
    rename = rename or {}
    if source_name in rename:
        return rename[source_name]
    match = re.search(r"(?:images_|depth_)?cam(?:era)?_?(\d+)$", source_name)
    if match:
        return f"cam_{match.group(1)}"
    clean = re.sub(r"[^a-zA-Z0-9_]+", "_", source_name).strip("_")
    return clean or source_name


def image_path(episode_dir: str, camera_dir: str, filename: str) -> str:
    return f"{episode_dir.rstrip('/')}/{camera_dir}/{filename}"


def read_rgb(source: zipfile.ZipFile | Path, path: str) -> np.ndarray:
    im = Image.open(io.BytesIO(read_member(source, path)))
    im = ImageOps.exif_transpose(im)
    if im.mode == "RGBA":
        bg = Image.new("RGBA", im.size, (0, 0, 0, 255))
        im = Image.alpha_composite(bg, im).convert("RGB")
    elif im.mode != "RGB":
        im = im.convert("RGB")
    arr = np.asarray(im, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"RGB image is not HWC uint8 RGB: {path} shape={arr.shape}")
    return np.ascontiguousarray(arr)


def read_depth(source: zipfile.ZipFile | Path, path: str) -> np.ndarray:
    im = Image.open(io.BytesIO(read_member(source, path)))
    arr = np.asarray(im)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.dtype not in (np.uint16, np.float32, np.float64, np.uint32, np.int32):
        raise ValueError(f"Unsupported depth dtype {arr.dtype}: {path}")
    return np.ascontiguousarray(arr)


class VectorStats:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.values: list[np.ndarray] = []

    def update(self, arr: np.ndarray) -> None:
        arr = np.asarray(arr, dtype=np.float64)
        if arr.shape != (self.dim,):
            raise ValueError(f"Expected vector dim {self.dim}, got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError("Vector stats received NaN/Inf")
        self.values.append(arr)

    def as_stats(self, quantiles: list[float]) -> dict[str, Any]:
        if not self.values:
            raise ValueError("No vector stats values")
        data = np.stack(self.values, axis=0)
        out: dict[str, Any] = {
            "min": data.min(axis=0).tolist(),
            "max": data.max(axis=0).tolist(),
            "mean": data.mean(axis=0).tolist(),
            "std": data.std(axis=0).tolist(),
            "count": [int(data.shape[0])] * data.shape[1],
        }
        for q in quantiles:
            out[quantile_name(q)] = np.quantile(data, q, axis=0).tolist()
        return out


class UintHistogramStats:
    def __init__(self, channels: int = 1, bins: int = 65536) -> None:
        self.channels = channels
        self.bins = bins
        self.hist = np.zeros((channels, bins), dtype=np.uint64)
        self.count = np.zeros(channels, dtype=np.uint64)
        self.sum = np.zeros(channels, dtype=np.float64)
        self.sumsq = np.zeros(channels, dtype=np.float64)

    def update(self, arr: np.ndarray) -> None:
        data = np.asarray(arr)
        if self.channels == 1:
            flat = data.reshape(-1)
            self._update_channel(0, flat)
            return
        if data.ndim != 3 or data.shape[2] != self.channels:
            raise ValueError(f"Expected HWC with {self.channels} channels, got {data.shape}")
        for c in range(self.channels):
            self._update_channel(c, data[..., c].reshape(-1))

    def _update_channel(self, c: int, flat: np.ndarray) -> None:
        flat = flat[np.isfinite(flat)] if np.issubdtype(flat.dtype, np.floating) else flat
        if flat.size == 0:
            return
        clipped = np.clip(flat.astype(np.int64), 0, self.bins - 1)
        self.hist[c] += np.bincount(clipped, minlength=self.bins).astype(np.uint64)
        self.count[c] += np.uint64(clipped.size)
        f = clipped.astype(np.float64)
        self.sum[c] += float(f.sum())
        self.sumsq[c] += float(np.square(f).sum())

    def as_stats(self, quantiles: list[float]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "min": [],
            "max": [],
            "mean": [],
            "std": [],
            "count": [int(x) for x in self.count],
        }
        for c in range(self.channels):
            count = int(self.count[c])
            if count == 0:
                raise ValueError("No histogram stats values")
            nonzero = np.flatnonzero(self.hist[c])
            out["min"].append(int(nonzero[0]))
            out["max"].append(int(nonzero[-1]))
            mean = float(self.sum[c] / count)
            var = max(float(self.sumsq[c] / count - mean * mean), 0.0)
            out["mean"].append(mean)
            out["std"].append(math.sqrt(var))
        for q in quantiles:
            out[quantile_name(q)] = [int(hist_quantile(self.hist[c], q)) for c in range(self.channels)]
        return out


def hist_quantile(hist: np.ndarray, q: float) -> int:
    total = int(hist.sum())
    if total <= 0:
        raise ValueError("Cannot compute quantile for empty histogram")
    rank = int(math.ceil(q * total))
    rank = max(rank, 1)
    return int(np.searchsorted(np.cumsum(hist), rank, side="left"))


def quantile_name(q: float) -> str:
    return f"q{int(round(q * 100)):02d}"


def merge_stats_into_lerobot(stats_path: Path, additions: dict[str, Any]) -> dict[str, Any]:
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    for key, value in additions.items():
        current = stats.get(key, {})
        if isinstance(current, dict):
            current.update(value)
            stats[key] = current
        else:
            stats[key] = value
    write_json(stats_path, stats)
    return stats
