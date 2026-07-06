from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from soft_vla.hardware.null_controller import NullRobotController
from soft_vla.hardware.safety_filter import SafetyFilter
from soft_vla.schemas import validate_action


@dataclass
class OfflineInferenceStats:
    frames: int
    mean_latency_ms: float
    mae: float
    rmse: float


def proportional_oracle(sample: dict) -> np.ndarray:
    return validate_action(np.asarray(sample["action"], dtype=np.float32))


def run_offline_inference(
    samples,
    policy: Callable[[dict], np.ndarray] = proportional_oracle,
    controller: NullRobotController | None = None,
    safety_filter: SafetyFilter | None = None,
    max_frames: int | None = None,
) -> tuple[OfflineInferenceStats, list[dict]]:
    controller = controller or NullRobotController()
    safety_filter = safety_filter or SafetyFilter()
    records: list[dict] = []
    latencies: list[float] = []
    errors: list[np.ndarray] = []

    for i, sample in enumerate(samples):
        if max_frames is not None and i >= max_frames:
            break
        gt = validate_action(np.asarray(sample["action"], dtype=np.float32))
        t0 = time.perf_counter()
        pred = validate_action(policy(sample))
        latency_ms = (time.perf_counter() - t0) * 1000.0
        filtered = safety_filter.filter_action(pred)
        controller.send_action(filtered)
        err = filtered - gt
        latencies.append(latency_ms)
        errors.append(err)
        records.append(
            {
                "frame": i,
                "latency_ms": latency_ms,
                "pred_action": filtered.tolist(),
                "gt_action": gt.tolist(),
                "abs_error": np.abs(err).tolist(),
            }
        )

    if errors:
        err_arr = np.stack(errors)
        mae = float(np.mean(np.abs(err_arr)))
        rmse = float(np.sqrt(np.mean(err_arr**2)))
    else:
        mae = 0.0
        rmse = 0.0
    stats = OfflineInferenceStats(
        frames=len(records),
        mean_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
        mae=mae,
        rmse=rmse,
    )
    return stats, records

