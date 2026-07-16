from __future__ import annotations

import unittest

import numpy as np

from soft_vla.human_intervention.target_integrator import HumanTargetIntegrator, HumanTargetIntegratorConfig


class HumanTargetIntegratorTest(unittest.TestCase):
    def test_same_xz_direction_integrates_and_clips(self):
        integ = HumanTargetIntegrator(HumanTargetIntegratorConfig(max_pos_offset=0.025, max_rot_offset=1.0))
        action = np.asarray([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        out1 = integ.step(action, active=True)
        out2 = integ.step(action, active=True)
        out3 = integ.step(action, active=True)
        self.assertAlmostEqual(float(out1.action7[0]), 0.01, places=7)
        self.assertAlmostEqual(float(out2.action7[0]), 0.02, places=7)
        self.assertAlmostEqual(float(out3.action7[0]), 0.025, places=7)
        self.assertEqual(out3.xz_direction, "x+")

    def test_direction_change_resets_xz_accumulation_to_current_delta(self):
        integ = HumanTargetIntegrator(HumanTargetIntegratorConfig(max_pos_offset=0.2, max_rot_offset=1.0))
        integ.step(np.asarray([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), active=True)
        integ.step(np.asarray([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), active=True)
        changed = integ.step(np.asarray([-0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), active=True)
        self.assertAlmostEqual(float(changed.action7[0]), -0.01, places=7)
        self.assertEqual(changed.xz_direction, "x-")

    def test_switching_from_x_to_z_drops_previous_x_offset(self):
        integ = HumanTargetIntegrator(HumanTargetIntegratorConfig(max_pos_offset=0.2, max_rot_offset=1.0))
        integ.step(np.asarray([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), active=True)
        integ.step(np.asarray([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), active=True)
        z = integ.step(np.asarray([0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), active=True)
        self.assertAlmostEqual(float(z.action7[0]), 0.0, places=7)
        self.assertAlmostEqual(float(z.action7[2]), 0.03, places=7)
        self.assertEqual(z.xz_direction, "z+")

    def test_inactive_resets_accumulation(self):
        integ = HumanTargetIntegrator(HumanTargetIntegratorConfig(max_pos_offset=0.2, max_rot_offset=1.0))
        integ.step(np.asarray([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), active=True)
        reset = integ.step(np.asarray([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), active=False)
        self.assertTrue(reset.reset)
        self.assertTrue(np.allclose(reset.accumulated_delta6, 0.0))
        restarted = integ.step(np.asarray([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), active=True)
        self.assertAlmostEqual(float(restarted.action7[0]), 0.01, places=7)


if __name__ == "__main__":
    unittest.main()
