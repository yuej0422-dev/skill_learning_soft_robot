from __future__ import annotations

import unittest

import numpy as np

from soft_vla.inference.chunk_execution.temporal_ensemble import TemporalEnsembleExecutor


class TemporalEnsembleTest(unittest.TestCase):
    def test_same_absolute_step_only(self):
        ex = TemporalEnsembleExecutor(weight_type="uniform")
        c0 = np.zeros((3, 7), dtype=np.float32)
        c1 = np.ones((3, 7), dtype=np.float32)
        ex.submit_chunk(c0, observation_timestamp=0, inference_start_timestamp=0, inference_end_timestamp=0)
        ex.submit_chunk(c1, observation_timestamp=1, inference_start_timestamp=0, inference_end_timestamp=0)
        rec = ex.get_action(1, 0.1)
        self.assertEqual(len(rec.debug["actions"]), 2)
        self.assertAlmostEqual(float(rec.action[0]), 0.5)
        rec2 = ex.get_action(3, 0.3)
        self.assertEqual(len(rec2.debug["actions"]), 1)
        self.assertAlmostEqual(float(rec2.action[0]), 1.0)

    def test_gripper_is_thresholded_after_weighted_vote(self):
        ex = TemporalEnsembleExecutor(weight_type="uniform")
        c0 = np.zeros((2, 7), dtype=np.float32)
        c1 = np.zeros((2, 7), dtype=np.float32)
        c1[:, 6] = 1.0
        ex.submit_chunk(c0, observation_timestamp=0, inference_start_timestamp=0, inference_end_timestamp=0)
        ex.submit_chunk(c1, observation_timestamp=0, inference_start_timestamp=0, inference_end_timestamp=0)
        rec = ex.get_action(0, 0.0)
        self.assertIn(float(rec.action[6]), (0.0, 1.0))
        self.assertEqual(float(rec.action[6]), 1.0)


if __name__ == "__main__":
    unittest.main()
