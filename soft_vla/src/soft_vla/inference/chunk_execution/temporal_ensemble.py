from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .base import ActionRecord, as_chunk, safe_fallback


@dataclass
class HistoricalChunk:
    chunk_id: int
    start_step: int
    actions: np.ndarray


class TemporalEnsembleExecutor:
    def __init__(
        self,
        *,
        replan_interval: int = 1,
        max_history_chunks: int = 10,
        weight_type: str = "exponential",
        decay: float = 0.25,
        prefer_newer_predictions: bool = True,
    ) -> None:
        self.replan_interval = replan_interval
        self.max_history_chunks = max_history_chunks
        self.weight_type = weight_type
        self.decay = decay
        self.prefer_newer_predictions = prefer_newer_predictions
        self.reset()

    def reset(self) -> None:
        self.history: list[HistoricalChunk] = []
        self.chunk_id = -1
        self.last_gripper = 0.0
        self.underruns = 0

    def submit_chunk(self, chunk, observation_timestamp: float, inference_start_timestamp: float, inference_end_timestamp: float) -> None:
        arr = as_chunk(chunk)
        self.chunk_id += 1
        start_step = int(round(observation_timestamp))
        self.history.append(HistoricalChunk(self.chunk_id, start_step, arr.copy()))
        self.history = self.history[-self.max_history_chunks :]
        self.timing = {
            "observation_timestamp": observation_timestamp,
            "inference_start_timestamp": inference_start_timestamp,
            "inference_end_timestamp": inference_end_timestamp,
        }

    def _weights(self, ages: np.ndarray) -> np.ndarray:
        if self.weight_type == "uniform":
            w = np.ones_like(ages, dtype=np.float64)
        elif self.weight_type == "linear":
            max_age = float(max(1, ages.max(initial=0) + 1))
            w = max_age - ages.astype(np.float64)
        elif self.weight_type == "exponential":
            w = np.exp(-float(self.decay) * ages.astype(np.float64))
        else:
            raise ValueError(f"unknown weight_type: {self.weight_type}")
        if not self.prefer_newer_predictions and self.weight_type == "exponential":
            w = np.exp(float(self.decay) * ages.astype(np.float64))
        return w / max(1e-12, w.sum())

    def get_action(self, control_step: int, control_timestamp: float) -> ActionRecord:
        candidates = []
        ages = []
        metadata = []
        for item in self.history:
            j = control_step - item.start_step
            if 0 <= j < len(item.actions):
                candidates.append(item.actions[j])
                ages.append(self.chunk_id - item.chunk_id)
                metadata.append({"chunk_id": item.chunk_id, "chunk_step": int(j)})
        if not candidates:
            self.underruns += 1
            return ActionRecord(safe_fallback(self.last_gripper), "te_underrun_fallback", None, None, control_step, 0, {})
        actions = np.stack(candidates).astype(np.float32)
        ages_arr = np.asarray(ages, dtype=np.int64)
        weights = self._weights(ages_arr).astype(np.float32)
        fused = np.sum(actions * weights[:, None], axis=0).astype(np.float32)
        self.last_gripper = float(fused[6])
        return ActionRecord(
            fused,
            "temporal_ensemble",
            metadata[-1]["chunk_id"],
            metadata[-1]["chunk_step"],
            control_step,
            int(ages_arr.min()) if len(ages_arr) else 0,
            {"weights": weights.tolist(), "ages": ages_arr.tolist(), "actions": actions.tolist(), "metadata": metadata},
        )

    def needs_replan(self, control_step: int, control_timestamp: float) -> bool:
        return control_step % self.replan_interval == 0

    def get_debug_state(self) -> dict:
        return {"mode": "temporal_ensemble", "history": len(self.history), "chunk_id": self.chunk_id, "underruns": self.underruns}

