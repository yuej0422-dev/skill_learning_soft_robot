from __future__ import annotations

from .fixed_chunk import FixedChunkExecutor
from .receding_horizon import RecedingHorizonExecutor
from .rtc_executor import RTCExecutor
from .single_step import SingleStepExecutor
from .temporal_ensemble import TemporalEnsembleExecutor


def make_chunk_executor(config: dict):
    mode = config.get("mode", "single_step")
    if mode == "single_step":
        return SingleStepExecutor()
    if mode == "chunk":
        return FixedChunkExecutor(
            chunk_size=int(config.get("chunk_size", 50)),
            execution_horizon=int(config.get("execution_horizon", 10)),
            expected_stale_steps=int(config.get("chunk_expected_stale_steps", config.get("expected_stale_steps", 2))),
            trigger_margin=int(config.get("chunk_trigger_margin", config.get("trigger_margin", 1))),
        )
    if mode == "receding_horizon":
        return RecedingHorizonExecutor(
            chunk_size=int(config.get("chunk_size", 50)),
            execution_horizon=int(config.get("execution_horizon", 5)),
            replan_interval=int(config.get("replan_interval", 5)),
            expected_stale_steps=int(config.get("chunk_expected_stale_steps", config.get("expected_stale_steps", 2))),
            trigger_margin=int(config.get("chunk_trigger_margin", config.get("trigger_margin", 1))),
        )
    if mode == "temporal_ensemble":
        return TemporalEnsembleExecutor(
            replan_interval=int(config.get("replan_interval", 1)),
            max_history_chunks=int(config.get("max_history_chunks", 10)),
            weight_type=str(config.get("weight_type", "exponential")),
            decay=float(config.get("decay", 0.25)),
            prefer_newer_predictions=bool(config.get("prefer_newer_predictions", True)),
        )
    if mode == "rtc":
        return RTCExecutor(**config.get("rtc", {}))
    raise ValueError(f"unknown chunk execution mode: {mode}")
