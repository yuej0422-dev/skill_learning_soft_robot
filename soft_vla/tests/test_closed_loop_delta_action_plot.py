from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from soft_vla.motion_control.controller_runtime import MotionControlRuntime
from soft_vla.real_robot.safety_manager import SafetyLimits, SafetyManager
from soft_vla.real_robot.single_point_plot import load_single_point_log


class _Feedforward:
    def predict(self, **kwargs):
        del kwargs
        return np.full(12, 0.4, dtype=np.float32)


class _Feedback:
    def predict(self, lifted_error):
        del lifted_error
        return np.linspace(-0.1, 0.1, 12, dtype=np.float32)


class ClosedLoopDeltaActionTest(unittest.TestCase):
    def test_runtime_exposes_feedback_as_closed_loop_delta_action(self):
        runtime = MotionControlRuntime(
            feedforward=_Feedforward(),
            feedback=_Feedback(),
            safety=SafetyManager(SafetyLimits(slew_rate_physical_per_s=None)),
        )
        command = runtime.compute(
            current_state12=np.zeros(12, dtype=np.float32),
            reference_state12=np.zeros(12, dtype=np.float32),
            delta_tcp6=np.zeros(6, dtype=np.float32),
            gripper_open=0.0,
            lifted_error=np.zeros(24, dtype=np.float64),
        )
        expected_delta = np.linspace(-0.1, 0.1, 12, dtype=np.float32)
        np.testing.assert_allclose(command.debug["feedforward_action12"], 0.4)
        np.testing.assert_allclose(command.debug["closed_loop_delta_action12"], expected_delta)
        np.testing.assert_allclose(command.debug["pre_safety_action12"], 0.4 + expected_delta)

    def test_plot_loader_reads_delta_action_and_keeps_legacy_logs_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.jsonl"
            row = {
                "step": 0,
                "measured_state": [0.0] * 12,
                "reference_state": [0.0] * 12,
                "motion_norm12": [0.2] * 12,
                "closed_loop_delta_action12": [0.05] * 12,
            }
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            data = load_single_point_log(path)
            np.testing.assert_allclose(data["closed_loop_delta_action12"], 0.05)

            del row["closed_loop_delta_action12"]
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            legacy = load_single_point_log(path)
            self.assertTrue(np.all(np.isnan(legacy["closed_loop_delta_action12"])))


if __name__ == "__main__":
    unittest.main()
