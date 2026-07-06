from __future__ import annotations

import unittest

import torch


class GripperLossWeightTest(unittest.TestCase):
    def test_weight_formula(self):
        losses = torch.ones(1, 2, 7)
        weights = torch.ones(7)
        weights[6] = 3.0
        weighted = (losses * weights.view(1, 1, -1)).mean()
        self.assertAlmostEqual(float(weighted), 9.0 / 7.0)


if __name__ == "__main__":
    unittest.main()

