from __future__ import annotations

from .fixed_chunk import FixedChunkExecutor


class RecedingHorizonExecutor(FixedChunkExecutor):
    def __init__(self, *, chunk_size: int = 50, execution_horizon: int = 5, replan_interval: int = 5) -> None:
        super().__init__(chunk_size=chunk_size, execution_horizon=execution_horizon)
        self.replan_interval = replan_interval
        self.last_submit_step = None
        self.boundary_records: list[dict] = []

    def submit_chunk(self, chunk, observation_timestamp: float, inference_start_timestamp: float, inference_end_timestamp: float) -> None:
        old_unexecuted = len(self.queue)
        old_last = self.queue[-1][0].copy().tolist() if self.queue else None
        super().submit_chunk(chunk, observation_timestamp, inference_start_timestamp, inference_end_timestamp)
        self.last_submit_step = int(round(observation_timestamp * 1e9)) if observation_timestamp < 1e-3 else None
        new_first = self.queue[0][0].copy().tolist() if self.queue else None
        self.boundary_records.append(
            {"old_unexecuted": old_unexecuted, "old_last_action": old_last, "new_first_action": new_first}
        )

    def needs_replan(self, control_step: int, control_timestamp: float) -> bool:
        return not self.queue or (control_step > 0 and control_step % self.replan_interval == 0)

    def get_debug_state(self) -> dict:
        out = super().get_debug_state()
        out.update({"mode": "receding_horizon", "replan_interval": self.replan_interval, "boundary_records": self.boundary_records[-10:]})
        return out

