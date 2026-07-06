from __future__ import annotations

import unittest

import numpy as np

from soft_vla.training.gripper import find_transition_indices, transition_window_mask


class GripperSamplerTest(unittest.TestCase):
    def test_transition_window(self):
        actions = np.zeros((8, 7), dtype=np.float32)
        actions[:, 6] = [0, 0, 1, 1, 1, 0, 0, 0]
        episodes = np.zeros(8, dtype=np.int64)
        transitions = find_transition_indices(actions, episodes)
        self.assertEqual(transitions.tolist(), [2, 5])
        mask, _ = transition_window_mask(actions, episodes, before_steps=1, after_steps=1)
        self.assertEqual(np.where(mask)[0].tolist(), [1, 2, 3, 4, 5, 6])


if __name__ == "__main__":
    unittest.main()

