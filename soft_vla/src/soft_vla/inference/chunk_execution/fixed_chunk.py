from __future__ import annotations

from collections import deque

import numpy as np

from .base import ActionRecord, as_chunk, safe_fallback


class FixedChunkExecutor:
    def __init__(self, *, chunk_size: int = 50, execution_horizon: int = 10) -> None:
        if execution_horizon > chunk_size:
            raise ValueError("execution_horizon must be <= chunk_size")
        self.chunk_size = chunk_size
        self.execution_horizon = execution_horizon
        self.reset()

    def reset(self) -> None:
        self.queue: deque[tuple[np.ndarray, int, int, int]] = deque()
        self.chunk_id = -1
        self.chunk_start_step = 0
        self.last_gripper = 0.0
        self.underruns = 0
        self.timing = {}

    def submit_chunk(self, chunk, observation_timestamp: float, inference_start_timestamp: float, inference_end_timestamp: float) -> None:
        arr = as_chunk(chunk)
        self.chunk_id += 1
        self.chunk_start_step = int(round(observation_timestamp))
        self.queue.clear()
        for j, action in enumerate(arr[: self.execution_horizon]):
            self.queue.append((action.copy(), self.chunk_id, j, self.chunk_start_step + j))
        self.timing = {
            "observation_timestamp": observation_timestamp,
            "inference_start_timestamp": inference_start_timestamp,
            "inference_end_timestamp": inference_end_timestamp,
        }

    def get_action(self, control_step: int, control_timestamp: float) -> ActionRecord:
        if not self.queue:
            self.underruns += 1
            return ActionRecord(safe_fallback(self.last_gripper), "queue_underrun_fallback", None, None, control_step, 0, {})
        action, chunk_id, chunk_step, rel_abs = self.queue.popleft()
        self.last_gripper = float(action[6])
        return ActionRecord(action, "chunk", chunk_id, chunk_step, rel_abs, control_step - rel_abs, dict(self.timing))

    def needs_replan(self, control_step: int, control_timestamp: float) -> bool:
        return not self.queue

    def get_debug_state(self) -> dict:
        return {"mode": "chunk", "queue_len": len(self.queue), "chunk_id": self.chunk_id, "underruns": self.underruns}
