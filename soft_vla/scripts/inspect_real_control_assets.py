from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()


def inspect_dataset(root: Path, *, episode_index: int = 0) -> dict:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        return {"error": f"pyarrow unavailable: {exc!r}"}
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    stats = json.loads((root / "meta/stats.json").read_text(encoding="utf-8"))
    data_file = next((root / "data").glob("chunk-*/file-*.parquet"))
    table = pq.read_table(data_file, columns=["episode_index", "frame_index", "timestamp", "observation.state", "action"])
    data = table.to_pydict()
    idx = [i for i, ep in enumerate(data["episode_index"]) if int(ep) == episode_index]
    timing = {}
    if idx:
        import numpy as np

        ts = np.asarray([data["timestamp"][i] for i in idx], dtype=float)
        frames = np.asarray([data["frame_index"][i] for i in idx], dtype=int)
        dt = np.diff(ts)
        timing = {
            "rows": int(len(idx)),
            "frame_min": int(frames.min()),
            "frame_max": int(frames.max()),
            "frame_contiguous": bool(np.array_equal(frames, np.arange(frames.size))),
            "dt_mean": float(dt.mean()) if dt.size else 0.0,
            "dt_min": float(dt.min()) if dt.size else 0.0,
            "dt_max": float(dt.max()) if dt.size else 0.0,
            "duplicate_timestamp": bool(len(set(ts.tolist())) != len(ts)),
        }
    raw_pressure_meta = json.loads((root / "meta/extra/raw_pressure_metadata.json").read_text(encoding="utf-8"))
    return {
        "root": str(root),
        "fps": info.get("fps"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "features": info.get("features", {}),
        "state_stats_dim": len(stats.get("observation.state", {}).get("mean", [])),
        "action_stats_dim": len(stats.get("action", {}).get("mean", [])),
        "raw_pressure_columns": raw_pressure_meta.get("columns", []),
        "episode_timing": timing,
    }


def inspect_torch_checkpoint(path: Path) -> dict:
    if path is None or not path.exists():
        return {"path": str(path), "exists": False}
    try:
        import torch
    except ImportError as exc:
        return {"path": str(path), "exists": True, "error": f"torch unavailable: {exc!r}"}
    ckpt = torch.load(path, map_location="cpu")
    out = {"path": str(path), "exists": True, "keys": sorted(ckpt.keys())}
    for key in ["layer_sizes", "state_indices", "normalization", "encode_layers", "n_koopman", "u_dim", "epoch", "best_val_loss"]:
        if key in ckpt:
            out[key] = ckpt[key]
    if "metadata" in ckpt:
        meta = ckpt["metadata"]
        out["metadata_summary"] = {
            "state_indices": meta.get("state_indices"),
            "pressure_indices": meta.get("pressure_indices"),
            "pressure_columns": meta.get("pressure_columns"),
            "target_offset": meta.get("target_offset"),
            "action_low": meta.get("action_low"),
            "action_high": meta.get("action_high"),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect real-control dataset and checkpoints without touching hardware.")
    parser.add_argument("--dataset-root", type=Path, default=Path("lerobot_conversion/outputs/robot_records_7_03_1_delta_tcp"))
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--pressure-checkpoint", type=Path, default=Path("motion_control_training/feedforward_pressure/runs/optimized_state12_raw_pressure/best.pt"))
    parser.add_argument("--awac-checkpoint", type=Path, default=Path("motion_control_training/KORL/runs/feedforward/awac_quadq_2k_eval_2x256/best.pt"))
    parser.add_argument(
        "--koopman-checkpoint",
        type=Path,
        default=Path("motion_control_training/koopman/runs/robot_records_7_03_1_delta_tcp_10hz_to_50hz_k50_epoch1500_wandb_online_20260706_2159/best.pt"),
    )
    args = parser.parse_args()
    report = {
        "dataset": inspect_dataset(args.dataset_root, episode_index=args.episode_index),
        "feedforward_pressure": inspect_torch_checkpoint(args.pressure_checkpoint),
        "awac_feedforward": inspect_torch_checkpoint(args.awac_checkpoint),
        "koopman": inspect_torch_checkpoint(args.koopman_checkpoint),
        "hardware_defaults": {"serial_port": "COM3", "baudrate": 115200, "packet_channels_default": 16},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
