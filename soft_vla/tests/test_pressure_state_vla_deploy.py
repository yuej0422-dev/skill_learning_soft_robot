from __future__ import annotations

import unittest

import numpy as np

from soft_vla.inference.chunk_execution.fixed_chunk import FixedChunkExecutor
from soft_vla.motion_control.controller_runtime import MotionControlRuntime
from soft_vla.motion_control.reference_generator import ReferenceGenerator, ReferenceGeneratorConfig
from soft_vla.real_robot.safety_manager import SafetyLimits, SafetyManager
from soft_vla.runtime.shared_state import UpperAction
from soft_vla.runtime.smolvla_async_runtime import (
    SmolVLAAsyncRuntimeConfig,
    _apply_inference_image_transforms,
    _build_vla_observation_state,
    _decode_vla_action,
    _inference_image_shapes,
    _preview_uses_exact_inference_input,
)


class _MustNotRunFeedforward:
    def predict(self, **kwargs):
        del kwargs
        raise AssertionError("the extra feedforward policy must be bypassed")


class _ConstantFeedback:
    def predict(self, lifted_error):
        del lifted_error
        return np.full(12, 0.05, dtype=np.float32)


class PressureStateVLADeployTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SmolVLAAsyncRuntimeConfig(
            vla_action_mode="pressure_delta19",
            pressure_delta_scale=0.5,
            feedforward="external",
            reference_interpolation="zero_order_hold",
        )
        self.latest_state = {
            "state12": np.arange(12, dtype=np.float32).tolist(),
            "gripper_open": 1.0,
            "motion_norm12": np.full(12, 0.4, dtype=np.float32).tolist(),
        }

    def test_state25_contains_current_normalized_pressure(self):
        state25 = _build_vla_observation_state(self.config, self.latest_state)
        self.assertEqual(state25.shape, (25,))
        np.testing.assert_allclose(state25[:12], np.arange(12, dtype=np.float32))
        self.assertEqual(float(state25[12]), 1.0)
        np.testing.assert_allclose(state25[13:], 0.4)

    def test_pressure_delta_is_added_to_current_pressure_and_clipped(self):
        action19 = np.zeros(19, dtype=np.float32)
        action19[:6] = 0.01
        action19[6] = 1.0
        action19[7:19] = np.asarray([0.2] * 6 + [2.0] * 3 + [-2.0] * 3, dtype=np.float32)
        action7, feedforward, pressure_delta = _decode_vla_action(self.config, action19, self.latest_state)
        np.testing.assert_allclose(action7, action19[:7])
        np.testing.assert_allclose(pressure_delta, action19[7:19])
        np.testing.assert_allclose(feedforward[:6], 0.5)
        np.testing.assert_allclose(feedforward[6:9], 1.0)
        np.testing.assert_allclose(feedforward[9:12], 0.0)

    def test_19d_chunk_executor_keeps_pressure_delta_dimensions(self):
        executor = FixedChunkExecutor(chunk_size=2, execution_horizon=1, action_dim=19)
        chunk = np.zeros((2, 19), dtype=np.float32)
        chunk[0, 6] = 1.0
        chunk[0, 7:] = 0.125
        executor.submit_chunk(chunk, 0.0, 0.0, 0.0)
        record = executor.get_action(0, 0.0)
        self.assertEqual(record.action.shape, (19,))
        np.testing.assert_allclose(record.action[7:], 0.125)
        fallback = executor.get_action(1, 0.1)
        self.assertEqual(fallback.action.shape, (19,))
        np.testing.assert_allclose(fallback.action[7:], 0.0)

    def test_10hz_target_is_zero_order_held_for_five_50hz_ticks(self):
        generator = ReferenceGenerator(
            ReferenceGeneratorConfig(
                upper_frequency_hz=10.0,
                control_frequency_hz=50.0,
                interpolation="zero_order_hold",
                delta_tcp_scale=1.0,
            )
        )
        current = np.zeros(12, dtype=np.float32)
        action = UpperAction(delta_tcp6=np.full(6, 0.01, dtype=np.float32), gripper_open=1.0, upper_step=0)
        segment = generator.build(current_state12=current, action=action)
        self.assertEqual(segment.reference_states12.shape, (5, 12))
        np.testing.assert_allclose(segment.reference_states12[:, :6], 0.01)
        for row in segment.reference_states12[1:]:
            np.testing.assert_allclose(row, segment.reference_states12[0])

    def test_external_vla_feedforward_replaces_extra_model_then_adds_feedback(self):
        runtime = MotionControlRuntime(
            feedforward=_MustNotRunFeedforward(),
            feedback=_ConstantFeedback(),
            safety=SafetyManager(SafetyLimits(slew_rate_physical_per_s=None)),
        )
        command = runtime.compute(
            current_state12=np.zeros(12, dtype=np.float32),
            reference_state12=np.zeros(12, dtype=np.float32),
            delta_tcp6=np.zeros(6, dtype=np.float32),
            gripper_open=1.0,
            lifted_error=np.zeros(24, dtype=np.float64),
            feedforward_action12=np.full(12, 0.4, dtype=np.float32),
        )
        np.testing.assert_allclose(command.debug["feedforward_action12"], 0.4)
        np.testing.assert_allclose(command.debug["closed_loop_delta_action12"], 0.05)
        np.testing.assert_allclose(command.motion_norm12, 0.45)

    def test_cam1_live_image_crop_matches_training_1280_to_1024(self):
        config = SmolVLAAsyncRuntimeConfig(cam1_crop_right_fraction=0.2)
        cam1 = np.arange(1280, dtype=np.uint16)[None, :, None]
        cam1 = np.broadcast_to(cam1, (720, 1280, 3)).copy()
        images = {
            "observation.images.cam_1": cam1,
            "observation.images.cam_2": np.zeros((480, 640, 3), dtype=np.uint8),
            "observation.images.cam_3": np.zeros((480, 640, 3), dtype=np.uint8),
        }
        transformed = _apply_inference_image_transforms(images, config)
        self.assertEqual(transformed["observation.images.cam_1"].shape, (720, 1024, 3))
        np.testing.assert_array_equal(transformed["observation.images.cam_1"], cam1[:, :1024])
        self.assertEqual(images["observation.images.cam_1"].shape, (720, 1280, 3))
        self.assertEqual(
            _inference_image_shapes(transformed)["observation.images.cam_1"],
            [720, 1024, 3],
        )

    def test_cam1_replay_tensor_uses_the_same_right_crop(self):
        import torch

        config = SmolVLAAsyncRuntimeConfig(cam1_crop_right_fraction=0.2)
        cam1 = torch.arange(1280, dtype=torch.float32).reshape(1, 1, 1280).expand(3, 720, 1280)
        transformed = _apply_inference_image_transforms({"observation.images.cam_1": cam1}, config)
        cropped = transformed["observation.images.cam_1"]
        self.assertEqual(tuple(cropped.shape), (3, 720, 1024))
        torch.testing.assert_close(cropped, cam1[..., :1024])

    def test_pressure_state_preview_is_driven_by_exact_inference_frames(self):
        self.assertTrue(
            _preview_uses_exact_inference_input(
                SmolVLAAsyncRuntimeConfig(vla_action_mode="pressure_delta19", cam1_crop_right_fraction=0.2)
            )
        )
        self.assertFalse(_preview_uses_exact_inference_input(SmolVLAAsyncRuntimeConfig()))


if __name__ == "__main__":
    unittest.main()
