from __future__ import annotations

from .base import ActionRecord, as_chunk


class SingleStepExecutor:
    def __init__(self, *, action_dim: int = 7) -> None:
        self.action_dim = int(action_dim)
        self.reset()

    def reset(self) -> None:
        self.chunk = None
        self.chunk_id = -1
        self.used = True

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
        self.chunk = as_chunk(chunk, action_dim=self.action_dim)
        self.chunk_id += 1
        request_tick = int(round(observation_timestamp)) if request_tick is None else int(request_tick)
        result_tick = request_tick if result_tick is None else int(result_tick)
        next_dispatch_tick = result_tick if next_dispatch_tick is None else int(next_dispatch_tick)
        effective_tick = max(result_tick, next_dispatch_tick)
        stale_steps = max(0, effective_tick - request_tick) if drop_stale_actions else 0
        self.local_idx = min(stale_steps, len(self.chunk) - 1)
        self.used = False
        self.timing = {
            "observation_timestamp": observation_timestamp,
            "inference_start_timestamp": inference_start_timestamp,
            "inference_end_timestamp": inference_end_timestamp,
            "request_tick": request_tick,
            "result_tick": result_tick,
            "next_dispatch_tick": next_dispatch_tick,
            "effective_tick": effective_tick,
            "stale_steps": stale_steps,
        }

    def get_action(self, control_step: int, control_timestamp: float) -> ActionRecord:
        if self.chunk is None or self.used:
            raise RuntimeError("single_step executor needs a fresh chunk before every action")
        self.used = True
        return ActionRecord(
            self.chunk[self.local_idx].copy(),
            "single_step",
            self.chunk_id,
            self.local_idx,
            control_step,
            control_step - int(self.timing["effective_tick"]),
            dict(self.timing),
        )

    def needs_replan(self, control_step: int, control_timestamp: float) -> bool:
        return self.chunk is None or self.used

    def get_debug_state(self) -> dict:
        return {"mode": "single_step", "used": self.used, "chunk_id": self.chunk_id}
