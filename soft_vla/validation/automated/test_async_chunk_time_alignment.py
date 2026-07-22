from __future__ import annotations

import unittest

import numpy as np

from soft_vla.inference.chunk_execution.fixed_chunk import FixedChunkExecutor
from soft_vla.inference.chunk_execution.receding_horizon import RecedingHorizonExecutor
from soft_vla.inference.chunk_execution.temporal_ensemble import TemporalEnsembleExecutor
from soft_vla.runtime.smolvla_async_runtime import SmolVLAAsyncRuntimeConfig, _result_tick_for_request


def fake_chunk(request_tick: int, size: int = 50) -> np.ndarray:
    chunk = np.zeros((size, 7), dtype=np.float32)
    chunk[:, 0] = request_tick * 1000 + np.arange(size, dtype=np.float32)
    chunk[:, 6] = 1.0
    return chunk


class AsyncChunkTimeAlignmentTest(unittest.TestCase):
    def test_chunk_drops_stale_actions(self):
        ex = FixedChunkExecutor(chunk_size=50, execution_horizon=10)
        ex.submit_chunk(
            fake_chunk(10),
            observation_timestamp=10,
            inference_start_timestamp=0,
            inference_end_timestamp=0,
            request_tick=10,
            result_tick=12,
            next_dispatch_tick=12,
        )
        rec = ex.get_action(12, 0.0)
        self.assertEqual(float(rec.action[0]), 10002.0)
        self.assertEqual(rec.chunk_step, 2)
        self.assertEqual(rec.debug["stale_steps"], 2)

    def test_chunk_drops_worst_case_stale_actions(self):
        ex = FixedChunkExecutor(chunk_size=50, execution_horizon=10)
        ex.submit_chunk(
            fake_chunk(10),
            observation_timestamp=10,
            inference_start_timestamp=0,
            inference_end_timestamp=0,
            request_tick=10,
            result_tick=15,
            next_dispatch_tick=15,
        )
        rec = ex.get_action(15, 0.0)
        self.assertEqual(float(rec.action[0]), 10005.0)
        self.assertEqual(rec.chunk_step, 5)
        self.assertEqual(rec.debug["stale_steps"], 5)

    def test_receding_horizon_updates_only_future_ticks(self):
        ex = RecedingHorizonExecutor(chunk_size=50, execution_horizon=10)
        ex.submit_chunk(
            fake_chunk(5),
            observation_timestamp=5,
            inference_start_timestamp=0,
            inference_end_timestamp=0,
            request_tick=5,
            result_tick=8,
            next_dispatch_tick=9,
        )
        rec = ex.get_action(9, 0.0)
        self.assertEqual(float(rec.action[0]), 5004.0)
        self.assertEqual(rec.absolute_step, 9)
        self.assertEqual(rec.chunk_step, 4)
        self.assertEqual(rec.debug["effective_tick"], 9)
        self.assertEqual(rec.debug["stale_steps"], 4)

    def test_temporal_ensemble_aligns_by_absolute_tick(self):
        ex = TemporalEnsembleExecutor(weight_type="uniform")
        c0 = np.zeros((50, 7), dtype=np.float32)
        c0[:, 0] = np.arange(50, dtype=np.float32)
        c0[:, 6] = 1.0
        c1 = fake_chunk(5)
        ex.submit_chunk(c0, observation_timestamp=0, inference_start_timestamp=0, inference_end_timestamp=0, request_tick=0)
        ex.submit_chunk(c1, observation_timestamp=5, inference_start_timestamp=0, inference_end_timestamp=0, request_tick=5)
        rec = ex.get_action(7, 0.0)
        self.assertEqual(rec.debug["te_candidate_chunk_ids"], [0, 1])
        self.assertEqual(rec.debug["te_candidate_local_indices"], [7, 2])
        self.assertAlmostEqual(float(rec.action[0]), (7.0 + 5002.0) / 2.0)

    def test_latency_to_result_tick_mapping(self):
        cfg = SmolVLAAsyncRuntimeConfig(upper_frequency_hz=10.0)
        request = {"request_tick": 10, "request_time_ns": 1_000_000_000}
        cases = [
            (160, 12),
            (180, 12),
            (405, 15),
        ]
        for latency_ms, expected_tick in cases:
            result_ns = request["request_time_ns"] + latency_ms * 1_000_000
            self.assertEqual(_result_tick_for_request(cfg, request, result_ns), expected_tick)

    def test_queue_underflow_returns_fallback_not_blocking(self):
        ex = FixedChunkExecutor(chunk_size=50, execution_horizon=1)
        ex.submit_chunk(
            fake_chunk(0, size=2),
            observation_timestamp=0,
            inference_start_timestamp=0,
            inference_end_timestamp=0,
            request_tick=0,
            result_tick=0,
            next_dispatch_tick=0,
        )
        self.assertEqual(float(ex.get_action(0, 0.0).action[0]), 0.0)
        rec = ex.get_action(1, 0.1)
        self.assertEqual(rec.source, "queue_underrun_fallback")
        self.assertEqual(float(rec.action[6]), 1.0)


if __name__ == "__main__":
    unittest.main()
