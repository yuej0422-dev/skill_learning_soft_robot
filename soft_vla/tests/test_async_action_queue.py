from __future__ import annotations

import unittest

import numpy as np

from soft_vla.inference.chunk_execution.fixed_chunk import FixedChunkExecutor


class AsyncActionQueueTest(unittest.TestCase):
    def test_queue_underrun_safe_action(self):
        ex = FixedChunkExecutor(chunk_size=2, execution_horizon=1)
        chunk = np.zeros((2, 7), dtype=np.float32)
        chunk[0, 6] = 1.0
        ex.submit_chunk(chunk, observation_timestamp=0, inference_start_timestamp=0, inference_end_timestamp=0)
        ex.get_action(0, 0)
        rec = ex.get_action(1, 0.1)
        self.assertEqual(rec.source, "queue_underrun_fallback")
        self.assertEqual(rec.action[:6].tolist(), [0.0] * 6)
        self.assertEqual(rec.action[6], 1.0)


if __name__ == "__main__":
    unittest.main()

