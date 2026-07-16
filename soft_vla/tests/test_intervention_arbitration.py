from __future__ import annotations

import unittest

import numpy as np

from soft_vla.human_intervention.action_mapper import HumanCommand
from soft_vla.human_intervention.intervention_manager import InterventionManager, InterventionManagerConfig


def cmd(action, *, active=True, connected=True, norm=1.0, success=False, failure=False, esc=False):
    return HumanCommand(
        action7=np.asarray(action, dtype=np.float32),
        active=active,
        input_norm=norm,
        success_pressed=success,
        failure_pressed=failure,
        esc_pressed=esc,
        gamepad_connected=connected,
    )


class InterventionManagerTest(unittest.TestCase):
    def test_vla_default_and_human_takeover(self):
        mgr = InterventionManager(InterventionManagerConfig(handover_blend_steps=0))
        vla = np.asarray([1, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        out = mgr.step(vla_action7=vla, human_command=None)
        self.assertEqual(out.action_source, "vla")
        self.assertEqual(float(out.executed_action7[0]), 1.0)

        human = cmd([0, 2, 0, 0, 0, 0, 0], active=True)
        out2 = mgr.step(vla_action7=vla, human_command=human)
        self.assertEqual(out2.action_source, "human")
        self.assertTrue(out2.intervention_active)
        self.assertEqual(float(out2.executed_action7[1]), 2.0)
        self.assertEqual(float(out2.executed_action7[6]), 0.0)

    def test_release_next_tick_uses_current_vla_without_fresh_wait(self):
        mgr = InterventionManager(InterventionManagerConfig(release_ticks=1, handover_blend_steps=0))
        mgr.step(vla_action7=np.asarray([1, 0, 0, 0, 0, 0, 1], dtype=np.float32), human_command=cmd([0, 2, 0, 0, 0, 0, 1]))
        out = mgr.step(
            vla_action7=np.asarray([3, 0, 0, 0, 0, 0, 1], dtype=np.float32),
            human_command=cmd([0, 0, 0, 0, 0, 0, 1], active=False, norm=0.0),
        )
        self.assertEqual(out.action_source, "vla")
        self.assertEqual(float(out.executed_action7[0]), 3.0)

    def test_vla_underflow_and_gamepad_disconnect_fallback(self):
        mgr = InterventionManager(InterventionManagerConfig(handover_blend_steps=0))
        out = mgr.step(vla_action7=None, human_command=None, vla_fallback=True)
        self.assertEqual(out.action_source, "fallback")
        mgr.step(vla_action7=np.ones(7, dtype=np.float32), human_command=cmd([0, 1, 0, 0, 0, 0, 1], active=True))
        human = cmd([0, 1, 0, 0, 0, 0, 1], active=True, connected=False)
        out2 = mgr.step(vla_action7=np.ones(7, dtype=np.float32), human_command=human)
        self.assertEqual(out2.action_source, "fallback")

    def test_handover_blending_tcp_only_and_gripper_discrete(self):
        mgr = InterventionManager(InterventionManagerConfig(handover_blend_steps=2, blend_gripper=False))
        mgr.step(vla_action7=np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32), human_command=None)
        mgr.step(vla_action7=np.zeros(7, dtype=np.float32), human_command=cmd([2, 0, 0, 0, 0, 0, 0]))
        out = mgr.step(
            vla_action7=np.asarray([4, 0, 0, 0, 0, 0, 1], dtype=np.float32),
            human_command=cmd([0, 0, 0, 0, 0, 0, 0], active=False, norm=0.0),
        )
        self.assertEqual(out.action_source, "vla")
        self.assertTrue(out.handover_blend_active)
        self.assertEqual(float(out.executed_action7[6]), 1.0)
        self.assertGreater(float(out.executed_action7[0]), 0.0)
        self.assertLess(float(out.executed_action7[0]), 4.0)

    def test_buttons_mark_episode_end(self):
        mgr = InterventionManager(InterventionManagerConfig(handover_blend_steps=0))
        out = mgr.step(vla_action7=np.zeros(7, dtype=np.float32), human_command=cmd([0, 0, 0, 0, 0, 0, 1], success=True))
        self.assertTrue(out.reset_triggered)
        self.assertEqual(out.termination_reason, "x_success")
        out2 = mgr.step(vla_action7=np.zeros(7, dtype=np.float32), human_command=cmd([0, 0, 0, 0, 0, 0, 1], failure=True))
        self.assertEqual(out2.termination_reason, "b_failure")

    def test_neutral_gripper_half_holds_previous_gripper_state(self):
        mgr = InterventionManager(InterventionManagerConfig(handover_blend_steps=0))
        mgr.step(vla_action7=np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32), human_command=None)
        closed = mgr.step(vla_action7=np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32), human_command=cmd([0, 0, 0, 0, 0, 0, 0]))
        self.assertEqual(closed.action_source, "human")
        self.assertEqual(float(closed.executed_action7[6]), 0.0)
        vla = np.asarray([0, 0, 0, 0, 0, 0, 0.5], dtype=np.float32)
        out = mgr.step(vla_action7=vla, human_command=cmd([0, 0, 0, 0, 0, 0, 0], active=False, norm=0.0))
        self.assertEqual(out.action_source, "vla")
        self.assertEqual(float(out.vla_action7[6]), 0.0)
        self.assertEqual(float(out.executed_action7[6]), 0.0)


if __name__ == "__main__":
    unittest.main()
