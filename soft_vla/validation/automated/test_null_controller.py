from __future__ import annotations

import unittest

import numpy as np

from soft_vla.hardware.null_controller import NullRobotController


class NullControllerTest(unittest.TestCase):
    def test_records_without_hardware(self):
        ctrl = NullRobotController()
        action = np.zeros(7, dtype=np.float32)
        ctrl.send_action(action)
        self.assertEqual(len(ctrl.recorded_actions), 1)


if __name__ == "__main__":
    unittest.main()

