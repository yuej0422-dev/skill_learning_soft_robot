from __future__ import annotations

from .base import ActionRecord, as_chunk


class SingleStepExecutor:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.chunk = None
        self.chunk_id = -1
        self.used = True

    def submit_chunk(self, chunk, observation_timestamp: float, inference_start_timestamp: float, inference_end_timestamp: float) -> None:
        self.chunk = as_chunk(chunk)
        self.chunk_id += 1
        self.used = False
        self.timing = {
            "observation_timestamp": observation_timestamp,
            "inference_start_timestamp": inference_start_timestamp,
            "inference_end_timestamp": inference_end_timestamp,
        }

    def get_action(self, control_step: int, control_timestamp: float) -> ActionRecord:
        if self.chunk is None or self.used:
            raise RuntimeError("single_step executor needs a fresh chunk before every action")
        self.used = True
        return ActionRecord(self.chunk[0].copy(), "single_step", self.chunk_id, 0, control_step, 0, dict(self.timing))

    def needs_replan(self, control_step: int, control_timestamp: float) -> bool:
        return self.chunk is None or self.used

    def get_debug_state(self) -> dict:
        return {"mode": "single_step", "used": self.used, "chunk_id": self.chunk_id}

