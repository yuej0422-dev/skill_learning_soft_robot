from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class RTCUnavailableError(RuntimeError):
    pass


@dataclass
class ActionRecord:
    action: np.ndarray
    source: str
    chunk_id: int | None
    chunk_step: int | None
    absolute_step: int | None
    action_age_steps: int
    debug: dict


class ChunkExecutor(Protocol):
    def reset(self) -> None: ...

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
    ) -> None: ...

    def get_action(self, control_step: int, control_timestamp: float) -> ActionRecord: ...

    def needs_replan(self, control_step: int, control_timestamp: float) -> bool: ...

    def get_debug_state(self) -> dict: ...


def as_chunk(chunk) -> np.ndarray:
    arr = np.asarray(chunk, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] != 7:
        raise ValueError(f"chunk must have shape [T,7] or [1,T,7], got {arr.shape}")
    return arr


def safe_fallback(last_gripper: float = 1.0) -> np.ndarray:
    action = np.zeros(7, dtype=np.float32)
    action[6] = 1.0 if last_gripper >= 0.5 else 0.0
    return action
