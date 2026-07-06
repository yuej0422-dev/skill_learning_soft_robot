from __future__ import annotations

import unittest

import numpy as np

from soft_vla.hardware.safety_filter import SafetyFilter


class SafetyFilterTest(unittest.TestCase):
    def test_clips_tcp_deltas(self):
        action = np.array([0.1, -0.1, 0.03, 0.2, -0.2, 0.1, 1.0], dtype=np.float32)
        filtered = SafetyFilter().filter_action(action)
        self.assertLessEqual(abs(filtered[0]), 0.02 + 1e-6)
        self.assertLessEqual(abs(filtered[3]), 0.08 + 1e-6)
        self.assertEqual(filtered[6], 1.0)


if __name__ == "__main__":
    unittest.main()

