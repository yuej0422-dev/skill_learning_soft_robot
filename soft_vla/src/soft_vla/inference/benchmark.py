from __future__ import annotations

from soft_vla.inference.runner import run_offline_inference


def benchmark(samples, max_frames: int | None = None):
    return run_offline_inference(samples, max_frames=max_frames)

