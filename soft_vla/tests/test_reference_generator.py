from __future__ import annotations

import unittest

import numpy as np

from soft_vla.motion_control.reference_generator import ReferenceGenerator, ReferenceGeneratorConfig, gripper_open_to_pressure4
from soft_vla.runtime.shared_state import UpperAction


class ReferenceGeneratorTest(unittest.TestCase):
    def test_linear_delta_is_split_across_five_substeps(self):
        gen = ReferenceGenerator(ReferenceGeneratorConfig(delta_tcp_scale=1.0, max_delta_tcp=None))
        current = np.zeros(12, dtype=np.float32)
        action = UpperAction(delta_tcp6=np.asarray([0.01, 0, 0, 0, 0, 0], dtype=np.float32), gripper_open=1, upper_step=2)
        segment = gen.build(current_state12=current, action=action)
        self.assertEqual(segment.reference_states12.shape, (5, 12))
        self.assertAlmostEqual(float(segment.reference_states12[0, 0]), 0.002)
        self.assertAlmostEqual(float(segment.reference_states12[-1, 0]), 0.01)
        self.assertEqual(segment.control_start_step, 10)

    def test_gripper_pressure_mapping(self):
        np.testing.assert_allclose(gripper_open_to_pressure4(1), np.asarray([3, 0, 0, 0], dtype=np.float32))
        np.testing.assert_allclose(gripper_open_to_pressure4(0), np.asarray([0, 3, 0, 0], dtype=np.float32))

    def test_delta_limit_applies_before_reference_generation(self):
        gen = ReferenceGenerator(
            ReferenceGeneratorConfig(delta_tcp_scale=1.0, max_delta_tcp=(0.01, 0.01, 0.01, 0.1, 0.1, 0.1))
        )
        action = UpperAction(delta_tcp6=np.ones(6, dtype=np.float32), gripper_open=0, upper_step=0)
        segment = gen.build(current_state12=np.zeros(12, dtype=np.float32), action=action)
        self.assertAlmostEqual(float(segment.reference_states12[-1, 0]), 0.01)
        self.assertAlmostEqual(float(segment.reference_states12[-1, 3]), 0.1)


if __name__ == "__main__":
    unittest.main()

