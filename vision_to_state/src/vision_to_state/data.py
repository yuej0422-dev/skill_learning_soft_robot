from __future__ import annotations

import csv
import io
import json
import math
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


STATE_NAMES = [
    "x",
    "y",
    "z",
    "rx",
    "ry",
    "rz",
    "vx",
    "vy",
    "vz",
    "wx",
    "wy",
    "wz",
]


@dataclass(frozen=True)
class RowItem:
    timestamp: float
    image_name: str
    depth_name: str
    state: np.ndarray


def record_dir_for_csv(csv_path: str) -> str:
    return csv_path.rsplit("/", 1)[0]


def read_csv_rows_from_zip(
    zf: zipfile.ZipFile,
    csv_path: str,
    image_column: int,
    depth_column: int,
    state_columns: str,
) -> list[RowItem]:
    rows: list[RowItem] = []
    with zf.open(csv_path) as f:
        reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
        next(reader, None)
        for raw in reader:
            if len(raw) < 17:
                continue
            if state_columns != "last12":
                raise ValueError("Only state_columns='last12' is currently supported.")
            if len(raw) < 12:
                continue
            try:
                state = np.asarray([float(x) for x in raw[-12:]], dtype=np.float32)
                timestamp = float(raw[0])
            except ValueError:
                continue
            if state.shape[0] != 12 or not np.isfinite(state).all():
                continue
            rows.append(
                RowItem(
                    timestamp=timestamp,
                    image_name=raw[image_column],
                    depth_name=raw[depth_column],
                    state=state,
                )
            )
    return rows


def contiguous_true_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(mask.tolist()):
        if value and start is None:
            start = i
        elif not value and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(mask)))
    return spans


def detect_linear_drop_spans(
    states: np.ndarray,
    window: int,
    residual_tol: float,
    min_disp: float,
) -> list[tuple[int, int]]:
    """Find near-perfect linear TCP-pose segments likely caused by state dropout."""
    n = states.shape[0]
    if n < window or window < 3:
        return []

    bad = np.zeros(n, dtype=bool)
    pose = states[:, :6].astype(np.float64)
    alpha = np.linspace(0.0, 1.0, window, dtype=np.float64)[:, None]
    for start in range(0, n - window + 1):
        seg = pose[start : start + window]
        disp = float(np.linalg.norm(seg[-1] - seg[0]))
        if disp < min_disp:
            continue
        linear = seg[0] + alpha * (seg[-1] - seg[0])
        residual = float(np.max(np.linalg.norm(seg - linear, axis=1)))
        if residual <= residual_tol:
            bad[start : start + window] = True
    return contiguous_true_spans(bad)


def analyze_record(rows: list[RowItem], cfg: dict[str, Any]) -> dict[str, Any]:
    min_rows = int(cfg["min_rows"])
    if len(rows) < min_rows:
        return {"status": "discard", "reason": "too_short", "n_rows": len(rows)}

    states = np.stack([r.state for r in rows], axis=0)
    state_range = states.max(axis=0) - states.min(axis=0)
    tcp_range_norm = float(np.linalg.norm(state_range[:6]))
    if tcp_range_norm <= float(cfg["min_tcp_range_norm"]):
        return {
            "status": "discard",
            "reason": "zero_or_constant_tcp",
            "n_rows": len(rows),
            "tcp_range_norm": tcp_range_norm,
            "state_range": state_range.tolist(),
        }

    spans = detect_linear_drop_spans(
        states,
        window=int(cfg["linearity_window"]),
        residual_tol=float(cfg["linearity_residual_tol"]),
        min_disp=float(cfg["linearity_min_disp"]),
    )
    linear_frames = sum(end - start for start, end in spans)
    linear_fraction = linear_frames / max(1, len(rows))
    if linear_fraction >= float(cfg["discard_record_if_linear_fraction_ge"]):
        status = "discard"
        reason = "linear_dropout"
    else:
        status = "keep"
        reason = "ok"

    return {
        "status": status,
        "reason": reason,
        "n_rows": len(rows),
        "tcp_range_norm": tcp_range_norm,
        "state_range": state_range.tolist(),
        "linear_spans": spans,
        "linear_fraction": linear_fraction,
    }


def build_manifest(config: dict[str, Any], max_records: int | None = None) -> dict[str, Any]:
    data_cfg = config["data"]
    zip_path = str(data_cfg["zip_path"])
    records = []
    with zipfile.ZipFile(zip_path) as zf:
        csv_paths = sorted(name for name in zf.namelist() if name.endswith("/data.csv"))
        if max_records is not None:
            csv_paths = csv_paths[:max_records]
        for csv_path in csv_paths:
            rows = read_csv_rows_from_zip(
                zf,
                csv_path,
                image_column=int(data_cfg["image_column"]),
                depth_column=int(data_cfg["depth_column"]),
                state_columns=str(data_cfg["state_columns"]),
            )
            stats = analyze_record(rows, data_cfg)
            n_windows = max(0, len(rows) - int(data_cfg["seq_len"]) + 1)
            stats.update(
                {
                    "csv_path": csv_path,
                    "record_dir": record_dir_for_csv(csv_path),
                    "n_windows_raw": n_windows,
                }
            )
            records.append(stats)

    summary: dict[str, Any] = {
        "total_records": len(records),
        "kept_records": sum(1 for r in records if r["status"] == "keep"),
        "discarded_records": sum(1 for r in records if r["status"] != "keep"),
        "discard_reasons": {},
    }
    for rec in records:
        if rec["status"] != "keep":
            reason = rec.get("reason", "unknown")
            summary["discard_reasons"][reason] = summary["discard_reasons"].get(reason, 0) + 1

    return {
        "source_zip": zip_path,
        "data_config": data_cfg,
        "records": records,
        "summary": summary,
    }


def save_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_or_build_manifest(config: dict[str, Any], rebuild: bool = False) -> dict[str, Any]:
    manifest_path = Path(config["data"]["manifest_path"])
    if manifest_path.exists() and not rebuild:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    manifest = build_manifest(config)
    save_manifest(manifest, manifest_path)
    return manifest


def split_records(
    records: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    kept = [r for r in records if r["status"] == "keep"]
    rng = random.Random(seed)
    rng.shuffle(kept)
    n = len(kept)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    return {
        "train": kept[:n_train],
        "val": kept[n_train : n_train + n_val],
        "test": kept[n_train + n_val :],
    }


def span_overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < span_end and end > span_start for span_start, span_end in spans)


class RGBDStateDataset(Dataset):
    def __init__(
        self,
        zip_path: str,
        records: list[dict[str, Any]],
        data_cfg: dict[str, Any],
        normalizer: dict[str, list[float]] | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.zip_path = zip_path
        self.records = records
        self.data_cfg = data_cfg
        self.seq_len = int(data_cfg["seq_len"])
        self.samples: list[tuple[int, int]] = []
        self.rows_by_record: list[list[RowItem]] = []
        self._zip: zipfile.ZipFile | None = None

        with zipfile.ZipFile(zip_path) as zf:
            for rec_idx, rec in enumerate(records):
                rows = read_csv_rows_from_zip(
                    zf,
                    rec["csv_path"],
                    image_column=int(data_cfg["image_column"]),
                    depth_column=int(data_cfg["depth_column"]),
                    state_columns=str(data_cfg["state_columns"]),
                )
                self.rows_by_record.append(rows)
                spans = [tuple(s) for s in rec.get("linear_spans", [])]
                for target_idx in range(self.seq_len - 1, len(rows)):
                    window_start = target_idx - self.seq_len + 1
                    if span_overlaps(window_start, target_idx + 1, spans):
                        continue
                    self.samples.append((rec_idx, target_idx))

        if max_samples is not None and len(self.samples) > max_samples:
            rng = random.Random(int(data_cfg["split_seed"]))
            self.samples = rng.sample(self.samples, max_samples)

        self.normalizer = normalizer

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_zip"] = None
        return state

    def _zf(self) -> zipfile.ZipFile:
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.zip_path)
        return self._zip

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgbd(self, rec: dict[str, Any], row: RowItem) -> torch.Tensor:
        zf = self._zf()
        base = rec["record_dir"].rstrip("/")
        camera_dir = str(self.data_cfg["camera_dir"]).strip("/")
        rgb_path = f"{base}/{camera_dir}/{row.image_name}"
        depth_path = f"{base}/{camera_dir}/{row.depth_name}"

        with zf.open(rgb_path) as f:
            rgb = Image.open(f).convert("RGB")
            rgb_arr = np.asarray(rgb)
        with zf.open(depth_path) as f:
            depth = Image.open(f)
            depth_arr = np.asarray(depth)

        if depth_arr.ndim == 3:
            depth_arr = depth_arr[..., 0]

        h, w = rgb_arr.shape[:2]
        x0 = int(round(w * float(self.data_cfg["crop_left_frac"])))
        y0 = int(round(h * float(self.data_cfg["crop_top_frac"])))
        rgb_arr = rgb_arr[y0:, x0:, :]
        depth_arr = depth_arr[y0:, x0:]

        image_size = int(self.data_cfg["image_size"])
        rgb_img = Image.fromarray(rgb_arr).resize((image_size, image_size), Image.BILINEAR)
        depth_img = Image.fromarray(depth_arr).resize((image_size, image_size), Image.NEAREST)
        rgb_arr = np.asarray(rgb_img, dtype=np.float32) / 255.0
        depth_arr = np.asarray(depth_img, dtype=np.float32)
        depth_arr = np.clip(depth_arr, 0.0, float(self.data_cfg["depth_clip_mm"]))
        depth_arr = depth_arr / float(self.data_cfg["depth_clip_mm"])

        rgb_chw = np.transpose(rgb_arr, (2, 0, 1))
        depth_chw = depth_arr[None, :, :]
        stacked = np.concatenate([rgb_chw, depth_chw], axis=0)
        return torch.from_numpy(stacked.astype(np.float32))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rec_idx, target_idx = self.samples[index]
        rec = self.records[rec_idx]
        rows = self.rows_by_record[rec_idx]
        start = target_idx - self.seq_len + 1
        frames = [self._load_rgbd(rec, rows[i]) for i in range(start, target_idx + 1)]
        image = torch.stack(frames, dim=0)
        target = torch.from_numpy(rows[target_idx].state.astype(np.float32))
        raw_target = target.clone()
        if self.normalizer is not None:
            mean = torch.tensor(self.normalizer["mean"], dtype=torch.float32)
            std = torch.tensor(self.normalizer["std"], dtype=torch.float32)
            target = (target - mean) / std
        return {"image": image, "target": target, "raw_target": raw_target}


def compute_normalizer(dataset: RGBDStateDataset) -> dict[str, list[float]]:
    labels = []
    for rec_idx, target_idx in dataset.samples:
        labels.append(dataset.rows_by_record[rec_idx][target_idx].state)
    arr = np.stack(labels, axis=0).astype(np.float64)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean.tolist(), "std": std.tolist(), "names": STATE_NAMES}


def rmse_mae(pred: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    err = pred - target
    rmse = np.sqrt(np.mean(err * err, axis=0))
    mae = np.mean(np.abs(err), axis=0)
    return {
        "rmse": rmse.tolist(),
        "mae": mae.tolist(),
        "pose_rmse_mean": float(np.mean(rmse[:6])),
        "vel_rmse_mean": float(np.mean(rmse[6:])),
        "pose_mae_mean": float(np.mean(mae[:6])),
        "vel_mae_mean": float(np.mean(mae[6:])),
    }


def pretty_metrics(metrics: dict[str, Any]) -> str:
    return (
        f"pose_rmse={metrics['pose_rmse_mean']:.6f} "
        f"vel_rmse={metrics['vel_rmse_mean']:.6f} "
        f"pose_mae={metrics['pose_mae_mean']:.6f} "
        f"vel_mae={metrics['vel_mae_mean']:.6f}"
    )
