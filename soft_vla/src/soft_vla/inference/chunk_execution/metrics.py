from __future__ import annotations

import numpy as np

from soft_vla.training.gripper import compute_gripper_metrics


def chunk_action_metrics(actions: np.ndarray, gt: np.ndarray | None = None, latencies_ms: list[float] | None = None) -> dict:
    actions = np.asarray(actions, dtype=np.float32)
    summary = {
        "frames": int(len(actions)),
        "translation_delta_norm_mean": float(np.linalg.norm(actions[:, :3], axis=1).mean()) if len(actions) else 0.0,
        "rotation_delta_norm_mean": float(np.linalg.norm(actions[:, 3:6], axis=1).mean()) if len(actions) else 0.0,
        "adjacent_action_diff_mean": float(np.abs(np.diff(actions, axis=0)).mean()) if len(actions) > 1 else 0.0,
        "discrete_jerk_l1_mean": float(np.abs(np.diff(actions, n=2, axis=0)).mean()) if len(actions) > 2 else 0.0,
        "gripper_switch_count": int(np.sum(np.diff((actions[:, 6] >= 0.5).astype(np.int32)) != 0)) if len(actions) > 1 else 0,
    }
    if latencies_ms:
        summary.update(
            {
                "mean_inference_latency_ms": float(np.mean(latencies_ms)),
                "p95_inference_latency_ms": float(np.percentile(latencies_ms, 95)),
            }
        )
    if gt is not None and len(actions):
        gt = np.asarray(gt, dtype=np.float32)[: len(actions)]
        err = actions - gt
        summary.update(
            {
                "overall_mae": float(np.mean(np.abs(err))),
                "per_dimension_mae": np.mean(np.abs(err), axis=0).tolist(),
                "tcp_overall_mae": float(np.mean(np.abs(err[:, :6]))),
                "gripper": compute_gripper_metrics(actions[:, 6], gt[:, 6]),
            }
        )
    return summary

