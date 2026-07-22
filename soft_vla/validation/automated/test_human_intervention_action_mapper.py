from __future__ import annotations

import unittest

import numpy as np

from soft_vla.human_intervention.action_mapper import HumanActionMapper, HumanActionMapperConfig


class HumanActionMapperTest(unittest.TestCase):
    def test_deadzone_and_shape(self):
        mapper = HumanActionMapper(HumanActionMapperConfig(joystick_deadzone=0.15))
        cmd = mapper.map_input({"axes": {"left_x": 0.1, "left_y": 0.0, "lt": 0.0, "rt": 0.0}, "buttons": {}})
        self.assertEqual(cmd.action7.shape, (7,))
        self.assertFalse(cmd.active)
        self.assertTrue(np.allclose(cmd.action7[:6], 0.0))

    def test_translation_limit_and_gripper_buttons(self):
        mapper = HumanActionMapper(
            HumanActionMapperConfig(max_delta_pos_per_tick=(0.001, 0.002, 0.003), max_action_slew_pos=0.01)
        )
        close_cmd = mapper.map_input({"axes": {"left_y": -1.0, "rt": 1.0}, "buttons": {"a": True}})
        self.assertAlmostEqual(float(close_cmd.action7[0]), -0.001, places=7)
        self.assertAlmostEqual(float(close_cmd.action7[1]), 0.002, places=7)
        self.assertAlmostEqual(float(close_cmd.action7[2]), 0.0, places=7)
        self.assertEqual(float(close_cmd.action7[6]), 0.0)
        self.assertEqual(close_cmd.gripper_command, 0)
        self.assertTrue(close_cmd.active)
        open_cmd = mapper.map_input({"axes": {}, "buttons": {"y": True}})
        self.assertEqual(float(open_cmd.action7[6]), 1.0)
        self.assertEqual(open_cmd.gripper_command, 1)

    def test_executed_gripper_sync_replaces_stale_human_latch(self):
        mapper = HumanActionMapper()
        mapper.map_input({"axes": {}, "buttons": {"a": True}})
        mapper.sync_gripper_state(1.0)

        neutral = mapper.map_input({"axes": {"left_y": -1.0}, "buttons": {}})
        self.assertIsNone(neutral.gripper_command)
        self.assertEqual(float(neutral.action7[6]), 1.0)

        close = mapper.map_input({"axes": {}, "buttons": {"a": True}})
        self.assertEqual(close.gripper_command, 0)
        self.assertEqual(float(close.action7[6]), 0.0)

    def test_rotation_disabled_and_single_axis_enabled(self):
        disabled = HumanActionMapper(HumanActionMapperConfig(rotation_enabled=False))
        cmd = disabled.map_input({"axes": {"right_x": 1.0, "right_y": 1.0}, "buttons": {}})
        self.assertTrue(np.allclose(cmd.action7[3:6], 0.0))

        yaw = HumanActionMapper(HumanActionMapperConfig(rotation_enabled=True, rotation_axis="yaw"))
        cmd2 = yaw.map_input({"axes": {"right_x": 1.0, "right_y": 0.0}, "buttons": {}})
        self.assertEqual(float(cmd2.action7[3]), 0.0)
        self.assertEqual(float(cmd2.action7[4]), 0.0)
        self.assertNotEqual(float(cmd2.action7[5]), 0.0)

    def test_pitch_yaw_rotation_uses_only_dominant_right_stick_axis(self):
        mapper = HumanActionMapper(
            HumanActionMapperConfig(rotation_enabled=True, rotation_axis="pitch_yaw", max_action_slew_rot=0.01)
        )
        yaw = mapper.map_input({"axes": {"right_x": 0.8, "right_y": 0.2}, "buttons": {}})
        self.assertEqual(float(yaw.action7[3]), 0.0)
        self.assertEqual(float(yaw.action7[4]), 0.0)
        self.assertNotEqual(float(yaw.action7[5]), 0.0)
        pitch = mapper.map_input({"axes": {"right_x": 0.2, "right_y": -0.8}, "buttons": {}})
        self.assertEqual(float(pitch.action7[3]), 0.0)
        self.assertNotEqual(float(pitch.action7[4]), 0.0)
        self.assertEqual(float(pitch.action7[5]), 0.0)

    def test_right_stick_pitch_sign_mapping(self):
        mapper = HumanActionMapper(
            HumanActionMapperConfig(
                rotation_enabled=True,
                rotation_axis="pitch_yaw",
                max_delta_rot_per_tick=(0.01, 0.01, 0.01),
                max_action_slew_rot=0.1,
            )
        )
        forward = mapper.map_input({"axes": {"right_y": -1.0}, "buttons": {}})
        self.assertLess(float(forward.action7[4]), 0.0)
        back = mapper.map_input({"axes": {"right_y": 1.0}, "buttons": {}})
        self.assertGreater(float(back.action7[4]), 0.0)

    def test_hysteresis_prevents_light_stick_jitter_takeover(self):
        mapper = HumanActionMapper(HumanActionMapperConfig(joystick_deadzone=0.15, intervention_release_deadzone=0.10))
        idle = mapper.map_input({"axes": {"left_x": 0.14}, "buttons": {}})
        self.assertFalse(idle.active)
        active = mapper.map_input({"axes": {"left_x": 0.2}, "buttons": {}})
        self.assertTrue(active.active)
        held = mapper.map_input({"axes": {"left_x": 0.12}, "buttons": {}})
        self.assertTrue(held.active)
        released = mapper.map_input({"axes": {"left_x": 0.05}, "buttons": {}})
        self.assertFalse(released.active)

    def test_translation_maps_left_stick_to_xz_and_triggers_to_vertical_y(self):
        mapper = HumanActionMapper(
            HumanActionMapperConfig(
                max_delta_pos_per_tick=(0.001, 0.002, 0.003),
                max_action_slew_pos=0.01,
            )
        )
        cmd = mapper.map_input({"axes": {"left_y": -1.0, "rt": 1.0, "lt": 0.0}, "buttons": {}})
        self.assertAlmostEqual(float(cmd.action7[0]), -0.001, places=7)
        self.assertAlmostEqual(float(cmd.action7[1]), 0.002, places=7)
        self.assertAlmostEqual(float(cmd.action7[2]), 0.0, places=7)

    def test_xz_plane_outputs_only_one_axis_at_a_time(self):
        mapper = HumanActionMapper(
            HumanActionMapperConfig(
                max_delta_pos_per_tick=(0.003, 0.002, 0.003),
                max_action_slew_pos=0.01,
            )
        )
        x_dominant = mapper.map_input({"axes": {"left_x": 0.5, "left_y": -1.0}, "buttons": {}})
        self.assertNotEqual(float(x_dominant.action7[0]), 0.0)
        self.assertEqual(float(x_dominant.action7[2]), 0.0)
        z_dominant = mapper.map_input({"axes": {"left_x": 1.0, "left_y": -0.2}, "buttons": {}})
        self.assertEqual(float(z_dominant.action7[0]), 0.0)
        self.assertNotEqual(float(z_dominant.action7[2]), 0.0)

    def test_left_stick_physical_direction_mapping(self):
        mapper = HumanActionMapper(
            HumanActionMapperConfig(
                max_delta_pos_per_tick=(0.01, 0.01, 0.01),
                max_action_slew_pos=0.1,
            )
        )
        left = mapper.map_input({"axes": {"left_x": -1.0}, "buttons": {}})
        self.assertLess(float(left.action7[2]), 0.0)
        right = mapper.map_input({"axes": {"left_x": 1.0}, "buttons": {}})
        self.assertGreater(float(right.action7[2]), 0.0)
        back = mapper.map_input({"axes": {"left_y": 1.0}, "buttons": {}})
        self.assertGreater(float(back.action7[0]), 0.0)
        forward = mapper.map_input({"axes": {"left_y": -1.0}, "buttons": {}})
        self.assertLess(float(forward.action7[0]), 0.0)

    def test_slew_limit_and_workspace_limit(self):
        mapper = HumanActionMapper(
            HumanActionMapperConfig(
                max_delta_pos_per_tick=(0.01, 0.01, 0.01),
                max_action_slew_pos=0.001,
                workspace_min=(0.0, 0.0, 0.0),
                workspace_max=(1.0, 1.0, 1.0),
            )
        )
        cmd = mapper.map_input(
            {"axes": {"left_y": 1.0}, "buttons": {}},
            current_state12=np.asarray([1.0, 0.5, 0.5] + [0.0] * 9, dtype=np.float32),
        )
        self.assertEqual(float(cmd.action7[0]), 0.0)
        self.assertIn("workspace_max_x", cmd.warnings)
        cmd2 = mapper.map_input({"axes": {"rt": 1.0}, "buttons": {}})
        self.assertLessEqual(abs(float(cmd2.action7[1])), 0.001 + 1e-8)


if __name__ == "__main__":
    unittest.main()
