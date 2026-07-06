from __future__ import annotations

import unittest

import numpy as np

from soft_vla.inference.chunk_execution.receding_horizon import RecedingHorizonExecutor


class RecedingHorizonExecutorTest(unittest.TestCase):
    def test_replan_interval(self):
        ex = RecedingHorizonExecutor(chunk_size=5, execution_horizon=3, replan_interval=3)
        chunk = np.zeros((5, 7), dtype=np.float32)
        ex.submit_chunk(chunk, observation_timestamp=0, inference_start_timestamp=0, inference_end_timestamp=0)
        self.assertFalse(ex.needs_replan(1, 0.1))
        self.assertTrue(ex.needs_replan(3, 0.3))


if __name__ == "__main__":
    unittest.main()

