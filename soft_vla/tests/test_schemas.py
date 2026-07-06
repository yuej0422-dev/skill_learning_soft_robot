from __future__ import annotations

import unittest

import numpy as np

from soft_vla.schemas import ACTION_DIM, STATE_DIM, validate_action, validate_state


class SchemaTest(unittest.TestCase):
    def test_state_and_action_accept_binary_gripper(self):
        state = np.zeros(STATE_DIM, dtype=np.float32)
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        state[12] = 1
        action[6] = 1
        self.assertEqual(validate_state(state).shape, (STATE_DIM,))
        self.assertEqual(validate_action(action).shape, (ACTION_DIM,))

    def test_reject_wrong_dims(self):
        with self.assertRaises(ValueError):
            validate_state(np.zeros(12, dtype=np.float32))
        with self.assertRaises(ValueError):
            validate_action(np.zeros(8, dtype=np.float32))

    def test_reject_nonbinary_gripper(self):
        state = np.zeros(STATE_DIM, dtype=np.float32)
        state[12] = 0.5
        with self.assertRaises(ValueError):
            validate_state(state)


if __name__ == "__main__":
    unittest.main()

