from __future__ import annotations

import unittest

import numpy as np

from soft_vla.inference.runner import run_offline_inference


class OfflinePipelineTest(unittest.TestCase):
    def test_runner(self):
        samples = [{"action": np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)} for _ in range(3)]
        stats, records = run_offline_inference(samples, max_frames=2)
        self.assertEqual(stats.frames, 2)
        self.assertEqual(len(records), 2)
        self.assertEqual(stats.mae, 0.0)


if __name__ == "__main__":
    unittest.main()

