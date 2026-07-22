from __future__ import annotations

import unittest

from soft_vla.training.gripper import apply_hybrid_action_stats


class HybridActionNormalizationTest(unittest.TestCase):
    def test_gripper_identity_stats(self):
        stats = {"action": {"mean": [10, 11, 12, 13, 14, 15, 0.7], "std": [1, 2, 3, 4, 5, 6, 0.2]}}
        patched = apply_hybrid_action_stats(stats)
        self.assertEqual(patched["action"]["mean"][:6], [10, 11, 12, 13, 14, 15])
        self.assertEqual(patched["action"]["std"][:6], [1, 2, 3, 4, 5, 6])
        self.assertEqual(patched["action"]["mean"][6], 0.0)
        self.assertEqual(patched["action"]["std"][6], 1.0)
        self.assertEqual(stats["action"]["mean"][6], 0.7)


if __name__ == "__main__":
    unittest.main()

