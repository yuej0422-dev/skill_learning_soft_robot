from __future__ import annotations

from collections import deque

import numpy as np

from .base import ActionRecord, as_chunk, safe_fallback


class FixedChunkExecutor:
    def __init__(
        self,
        *,
        chunk_size: int = 50,
        execution_horizon: int = 10,
        expected_stale_steps: int = 0,
        trigger_margin: int = 0,
        action_dim: int = 7,
    ) -> None:
        if execution_horizon > chunk_size:
            raise ValueError("execution_horizon must be <= chunk_size")
        self.chunk_size = chunk_size
        self.execution_horizon = execution_horizon
        self.expected_stale_steps = max(0, int(expected_stale_steps))
        self.trigger_margin = max(0, int(trigger_margin))
        self.action_dim = int(action_dim)
        self.reset()

    def reset(self) -> None:
        self.queue: deque[tuple[np.ndarray, int, int, int]] = deque()
        self.chunk_id = -1
        self.chunk_start_step = 0
        self.last_gripper = 1.0
        self.underruns = 0
        self.timing = {}

    def submit_chunk(
        self,
        chunk,
        observation_timestamp: float,
        inference_start_timestamp: float,
        inference_end_timestamp: float,
        *,
        request_tick: int | None = None,
        result_tick: int | None = None,
        next_dispatch_tick: int | None = None,
        drop_stale_actions: bool = True,
    ) -> None:
        arr = as_chunk(chunk, action_dim=self.action_dim)
        self.chunk_id += 1
        request_tick = int(round(observation_timestamp)) if request_tick is None else int(request_tick)
        result_tick = request_tick if result_tick is None else int(result_tick)
        next_dispatch_tick = result_tick if next_dispatch_tick is None else int(next_dispatch_tick)
        effective_tick = max(result_tick, next_dispatch_tick)
        stale_steps = max(0, effective_tick - request_tick) if drop_stale_actions else 0
        valid = arr[stale_steps:]
        self.chunk_start_step = effective_tick
        self.queue.clear()
        for j, action in enumerate(valid[: self.execution_horizon]):
            local_idx = stale_steps + j
            self.queue.append((action.copy(), self.chunk_id, local_idx, effective_tick + j))
        self.timing = {
            "observation_timestamp": observation_timestamp,
            "inference_start_timestamp": inference_start_timestamp,
            "inference_end_timestamp": inference_end_timestamp,
            "request_tick": request_tick,
            "result_tick": result_tick,
            "next_dispatch_tick": next_dispatch_tick,
            "effective_tick": effective_tick,
            "valid_start_tick": effective_tick,
            "stale_steps": stale_steps,
            "dropped_stale_actions": stale_steps,
            "selected_actions": len(self.queue),
        }

    def get_action(self, control_step: int, control_timestamp: float) -> ActionRecord:
        if not self.queue:
            self.underruns += 1
            return ActionRecord(
                safe_fallback(self.last_gripper, action_dim=self.action_dim),
                "queue_underrun_fallback",
                None,
                None,
                control_step,
                0,
                {},
            )
        action, chunk_id, chunk_step, rel_abs = self.queue.popleft()
        self.last_gripper = float(action[6])
        return ActionRecord(action, "chunk", chunk_id, chunk_step, rel_abs, control_step - rel_abs, dict(self.timing))

    def needs_replan(self, control_step: int, control_timestamp: float) -> bool:
        threshold = self.expected_stale_steps + self.trigger_margin
        return not self.queue or len(self.queue) <= threshold

    def get_debug_state(self) -> dict:
        return {
            "mode": "chunk",
            "queue_len": len(self.queue),
            "chunk_id": self.chunk_id,
            "underruns": self.underruns,
            "expected_stale_steps": self.expected_stale_steps,
            "trigger_margin": self.trigger_margin,
            "action_dim": self.action_dim,
            "timing": dict(self.timing),
        }
