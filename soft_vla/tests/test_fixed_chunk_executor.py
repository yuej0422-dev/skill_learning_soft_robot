from __future__ import annotations

import unittest

import numpy as np

from soft_vla.inference.chunk_execution.fixed_chunk import FixedChunkExecutor


class FixedChunkExecutorTest(unittest.TestCase):
    def test_executes_horizon_then_fallback(self):
        ex = FixedChunkExecutor(chunk_size=5, execution_horizon=2)
        chunk = np.zeros((5, 7), dtype=np.float32)
        chunk[:, 0] = np.arange(5)
        chunk[:, 6] = 1
        ex.submit_chunk(chunk, observation_timestamp=10, inference_start_timestamp=0, inference_end_timestamp=0)
        self.assertEqual(ex.get_action(10, 1.0).absolute_step, 10)
        self.assertEqual(ex.get_action(11, 1.1).chunk_step, 1)
        self.assertTrue(ex.needs_replan(12, 1.2))
        rec = ex.get_action(12, 1.2)
        self.assertEqual(rec.source, "queue_underrun_fallback")
        self.assertEqual(rec.action[6], 1.0)


if __name__ == "__main__":
    unittest.main()

