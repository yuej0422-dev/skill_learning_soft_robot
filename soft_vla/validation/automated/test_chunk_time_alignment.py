from __future__ import annotations

import unittest

import numpy as np

from soft_vla.inference.chunk_execution.fixed_chunk import FixedChunkExecutor


class ChunkTimeAlignmentTest(unittest.TestCase):
    def test_absolute_step_alignment(self):
        ex = FixedChunkExecutor(chunk_size=4, execution_horizon=4)
        ex.submit_chunk(np.zeros((4, 7), dtype=np.float32), observation_timestamp=7, inference_start_timestamp=0, inference_end_timestamp=0)
        self.assertEqual(ex.get_action(7, 0).absolute_step, 7)
        self.assertEqual(ex.get_action(8, 0).absolute_step, 8)


if __name__ == "__main__":
    unittest.main()

