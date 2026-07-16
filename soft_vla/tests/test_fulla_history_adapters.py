from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from soft_vla.motion_control.fulla_history_adapters import (
    FullAHistoryKoopmanAdapter,
    FullAHistoryKoopmanConfig,
)
from soft_vla.motion_control.feedforward_adapters import FeedforwardPressureConfig, FeedforwardPressureMLPAdapter


ROOT = Path(__file__).resolve().parents[2]
PRESSURE_CHECKPOINT = (
    ROOT / "motion_control_training/feedforward_pressure/runs/tcp6_target_raw_pressure/best.pt"
)
KOOPMAN_CHECKPOINT = ROOT / (
    "motion_control_training/koopman/experiments/fullA_history_v2/runs/"
    "robot_records_7_03_1_delta_tcp_fullA_history_v2_50hz_k50_hist10_epoch3000_wandb_online_20260712_2320/"
    "best.pt"
)


@unittest.skipUnless(PRESSURE_CHECKPOINT.exists() and KOOPMAN_CHECKPOINT.exists(), "local model assets are unavailable")
class FullAHistoryDeploymentAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.koopman = FullAHistoryKoopmanAdapter(
            FullAHistoryKoopmanConfig(checkpoint=KOOPMAN_CHECKPOINT, device="cpu")
        )

    def setUp(self) -> None:
        self.koopman.reset()

    def test_checkpoint_contract_and_zero_tracking_error(self):
        self.assertEqual(self.koopman.n_koopman, 24)
        self.assertEqual(self.koopman.history_steps, 10)
        self.assertEqual(self.koopman.target_hz, 50.0)
        state = self.koopman.state_mean.copy()
        error = self.koopman.tracking_error(state, state)
        np.testing.assert_allclose(error, np.zeros(24), atol=1e-7)
        self.koopman.record_control(np.zeros(12, dtype=np.float32))

    def test_actual_normalized_pressure_is_recorded_in_history(self):
        state = self.koopman.state_mean.copy()
        self.koopman.tracking_error(state, state)
        pressure = np.linspace(0.0, 0.9, 12, dtype=np.float32)
        self.koopman.record_control(pressure)
        snapshot = self.koopman.history_snapshot()
        np.testing.assert_allclose(snapshot["normalized_pressure"][-1], pressure)
        self.assertEqual(snapshot["normalized_pressure"].shape, (10, 12))

    def test_requires_one_control_per_tracking_step(self):
        state = self.koopman.state_mean.copy()
        self.koopman.tracking_error(state, state)
        with self.assertRaises(RuntimeError):
            self.koopman.tracking_error(state, state)
        self.koopman.record_control(np.zeros(12, dtype=np.float32))

    def test_feedforward_checkpoint_outputs_normalized_pressure(self):
        adapter = FeedforwardPressureMLPAdapter(
            FeedforwardPressureConfig(checkpoint=PRESSURE_CHECKPOINT, device="cpu", input_mode="target_state")
        )
        reference = np.asarray(
            [0.060958, 0.648326, 0.072231, 0.059275, 0.019778, 0.013621] + [0.0] * 6,
            dtype=np.float32,
        )
        prediction = adapter.predict(
            current_state12=reference,
            reference_state12=reference,
            delta_tcp6=np.zeros(6, dtype=np.float32),
        )
        self.assertEqual(prediction.shape, (12,))
        self.assertTrue(np.all(prediction >= 0.0))
        self.assertTrue(np.all(prediction <= 1.0))

    def test_rejects_physical_bar_values_in_koopman_history(self):
        state = self.koopman.state_mean.copy()
        self.koopman.tracking_error(state, state)
        with self.assertRaises(ValueError):
            self.koopman.record_control(np.full(12, 3.0, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
