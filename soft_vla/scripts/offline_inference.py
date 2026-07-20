from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _bootstrap import add_src_to_path

PROJECT_ROOT = add_src_to_path()

from soft_vla.config import load_yaml
from soft_vla.data.replay_source import LeRobotReplaySource
from soft_vla.inference.runner import run_offline_inference


def apply_image_transforms(batch: dict, cfg: dict) -> dict:
    transforms = cfg.get("image_transforms", {})
    crop_cfg = transforms.get("crop_right_fraction", {})
    if not crop_cfg:
        return batch
    out = dict(batch)
    for key, fraction in crop_cfg.items():
        if key not in out:
            continue
        value = out[key]
        if not hasattr(value, "shape") or value.ndim < 3:
            continue
        width = int(value.shape[-1])
        keep_width = int(round(width * (1.0 - float(fraction))))
        if keep_width <= 0 or keep_width > width:
            raise ValueError(f"Invalid crop_right_fraction={fraction} for {key} width={width}")
        out[key] = value[..., :keep_width].contiguous()
    return out


def validate_vector(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        raise ValueError(f"{name} must have a trailing dimension.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf.")
    return arr


def dry_run_safety_filter(action: np.ndarray) -> np.ndarray:
    arr = validate_vector(action, "predicted action").copy()
    if arr.shape[-1] >= 6:
        arr[..., :3] = np.clip(arr[..., :3], -0.02, 0.02)
        arr[..., 3:6] = np.clip(arr[..., 3:6], -0.08, 0.08)
    if arr.shape[-1] >= 7:
        arr[..., 6] = np.where(arr[..., 6] >= 0.5, 1.0, 0.0)
    return arr.astype(np.float32)


def apply_sigmoid_gripper_np(action: np.ndarray, *, gripper_index: int = 6) -> np.ndarray:
    arr = validate_vector(action, "predicted action").copy()
    arr[..., gripper_index] = 1.0 / (1.0 + np.exp(-arr[..., gripper_index]))
    return arr.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--policy-type", default=None, choices=["smolvla", "oracle_baseline"])
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    cfg = load_yaml(PROJECT_ROOT / args.config)
    ds_cfg_path = PROJECT_ROOT / cfg["dataset"]["config"]
    ds_cfg = load_yaml(ds_cfg_path)["dataset"]
    root = Path(cfg["dataset"].get("root") or ds_cfg["root"])
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    source = LeRobotReplaySource(
        root,
        ds_cfg.get("repo_id"),
        cfg["dataset"].get("episode_index"),
        video_backend=cfg["dataset"].get("video_backend", ds_cfg.get("video_backend")),
    )
    policy_type = args.policy_type or cfg["policy"].get("policy_type", "smolvla")
    out_dir = Path(cfg["inference"]["output_dir"])
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if policy_type == "oracle_baseline":
        stats, records = run_offline_inference(source, max_frames=cfg["inference"].get("max_frames"))
        (out_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        (out_dir / "stats.json").write_text(json.dumps({**stats.__dict__, "policy_type": "oracle_baseline"}, indent=2), encoding="utf-8")
        print(json.dumps({**stats.__dict__, "policy_type": "oracle_baseline"}, indent=2))
        return 0

    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    checkpoint = Path(args.checkpoint or cfg["policy"]["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    device = cfg["policy"].get("device", "cuda")
    policy = SmolVLAPolicy.from_pretrained(checkpoint, local_files_only=True)
    policy.config.device = device
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    max_frames = cfg["inference"].get("max_frames")
    use_safety_filter = bool(cfg["inference"].get("safety_filter", True))
    use_sigmoid_gripper_head = bool(cfg.get("gripper_action_head", {}).get("sigmoid_bounded", False))
    records = []
    latencies = []
    errors = []
    first_chunk = None
    first_raw = None
    first_processed = None
    torch.cuda.reset_peak_memory_stats()
    policy.reset()
    for i, sample in enumerate(source):
        if max_frames is not None and i >= max_frames:
            break
        gt = validate_vector(sample["action"], "ground-truth action")
        obs = {k: v for k, v in sample.items() if k != "action" and k != "action_is_pad"}
        obs = apply_image_transforms(obs, cfg)
        if first_raw is None:
            first_raw = {
                "keys": list(obs.keys()),
                "state": np.asarray(obs["observation.state"]).tolist(),
                "image_shapes": {
                    k: list(v.shape)
                    for k, v in obs.items()
                    if k.startswith("observation.images") and hasattr(v, "shape")
                },
                "task": str(obs.get("task")),
            }
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        batch = preprocessor(obs)
        if first_processed is None:
            first_processed = {
                "keys": list(batch.keys()),
                "state_shape": list(batch["observation.state"].shape),
                "state": batch["observation.state"].detach().cpu().tolist(),
                "image_shapes": {
                    k: list(v.shape)
                    for k, v in batch.items()
                    if k.startswith("observation.images") and hasattr(v, "shape")
                },
            }
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
            action_chunk = policy.predict_action_chunk(batch)
        raw_chunk = postprocessor.process_action(action_chunk).detach().cpu()
        if use_sigmoid_gripper_head:
            raw_chunk = torch.as_tensor(apply_sigmoid_gripper_np(raw_chunk.numpy()))
        t1.record()
        torch.cuda.synchronize()
        latency_ms = float(t0.elapsed_time(t1))
        pred = raw_chunk[0, 0].numpy().astype(np.float32)
        if use_safety_filter and not use_sigmoid_gripper_head:
            pred = dry_run_safety_filter(pred)
        elif use_safety_filter:
            pred = validate_vector(pred, "predicted action").copy()
            pred[:3] = np.clip(pred[:3], -0.02, 0.02)
            pred[3:6] = np.clip(pred[3:6], -0.08, 0.08)
        if pred.shape != gt.shape:
            raise ValueError(f"predicted action shape {pred.shape} does not match ground-truth shape {gt.shape}")
        err = pred - gt
        latencies.append(latency_ms)
        errors.append(err)
        if first_chunk is None:
            first_chunk = raw_chunk.numpy()
        records.append(
            {
                "frame": i,
                "latency_ms": latency_ms,
                "pred_action_shape": list(pred.shape),
                "gt_action_shape": list(gt.shape),
                "pred_action": pred.tolist(),
                "gt_action": gt.tolist(),
                "abs_error": np.abs(err).tolist(),
            }
        )
    err_arr = np.stack(errors) if errors else np.zeros((0, 7), dtype=np.float32)
    summary = {
        "checkpoint": str(checkpoint),
        "policy_type": "smolvla",
        "episode_index": cfg["dataset"].get("episode_index"),
        "frames": len(records),
        "action_chunk_shape": list(first_chunk.shape) if first_chunk is not None else None,
        "raw_action_shape": list(records[0]["pred_action_shape"]) if records else None,
        "gt_action_shape": list(records[0]["gt_action_shape"]) if records else None,
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "median_latency_ms": float(np.median(latencies)) if latencies else 0.0,
        "p90_latency_ms": float(np.percentile(latencies, 90)) if latencies else 0.0,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "overall_mae": float(np.mean(np.abs(err_arr))) if len(err_arr) else 0.0,
        "overall_rmse": float(np.sqrt(np.mean(err_arr**2))) if len(err_arr) else 0.0,
        "per_dimension_mae": np.mean(np.abs(err_arr), axis=0).tolist() if len(err_arr) else [],
        "gripper_prediction_values": sorted(set(float(r["pred_action"][6]) for r in records)) if records and len(records[0]["pred_action"]) > 6 else [],
        "safety_filter": use_safety_filter,
        "gripper_action_head": {
            "sigmoid_bounded": use_sigmoid_gripper_head,
            "postprocess": "sigmoid(action[6])" if use_sigmoid_gripper_head else "none",
        },
        "dry_run": True,
    }
    (out_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "first_sample_raw.json").write_text(json.dumps(first_raw, indent=2), encoding="utf-8")
    (out_dir / "first_sample_processed.json").write_text(json.dumps(first_processed, indent=2), encoding="utf-8")
    if first_chunk is not None:
        np.save(out_dir / "first_action_chunk.npy", first_chunk)
    report_name = cfg["inference"].get("report_name", "offline_inference_smolvla.md")
    (PROJECT_ROOT / "reports" / report_name).write_text(
        "# Offline Inference SmolVLA\n\n"
        f"- Checkpoint: `{checkpoint}`\n"
        "- Policy type: `smolvla`\n"
        f"- Episode index: `{summary['episode_index']}`\n"
        f"- Frames: `{summary['frames']}`\n"
        f"- Action chunk shape: `{summary['action_chunk_shape']}`\n"
        f"- Mean latency ms: `{summary['mean_latency_ms']}`\n"
        f"- P95 latency ms: `{summary['p95_latency_ms']}`\n"
        f"- Peak GPU memory GiB: `{summary['peak_gpu_memory_gb']}`\n"
        f"- Overall MAE: `{summary['overall_mae']}`\n"
        f"- Overall RMSE: `{summary['overall_rmse']}`\n"
        f"- Per-dimension MAE: `{summary['per_dimension_mae']}`\n"
        f"- Gripper prediction values after safety filter: `{summary['gripper_prediction_values']}`\n"
        "\nThis is offline action fitting error on replay data, not real task success rate.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
