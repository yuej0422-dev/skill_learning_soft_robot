from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soft_vla.motion_control.feedforward_adapters import FeedforwardPolicy, ZeroFeedforwardPolicy
from soft_vla.motion_control.feedback_controllers import IntegralFeedbackController
from soft_vla.real_robot.safety_manager import SafetyManager
from soft_vla.runtime.shared_state import PressureCommand


@dataclass
class MotionControlRuntime:
    feedforward: FeedforwardPolicy | None
    feedback: IntegralFeedbackController | None
    safety: SafetyManager

    def __post_init__(self) -> None:
        if self.feedforward is None:
            self.feedforward = ZeroFeedforwardPolicy()

    def compute(
        self,
        *,
        current_state12: np.ndarray,
        reference_state12: np.ndarray,
        delta_tcp6: np.ndarray,
        gripper_open: float,
        lifted_error: np.ndarray | None = None,
        now_ns: int | None = None,
        state_timestamp_ns: int | None = None,
        reference_timestamp_ns: int | None = None,
        pressure_scale: float = 1.0,
    ) -> PressureCommand:
        ff = self.feedforward.predict(
            current_state12=current_state12,
            reference_state12=reference_state12,
            delta_tcp6=delta_tcp6,
        )
        feedback = np.zeros(12, dtype=np.float32)
        if self.feedback is not None:
            if lifted_error is None:
                raise ValueError("lifted_error is required when feedback controller is enabled")
            feedback = self.feedback.predict(lifted_error)
        motion_norm = np.asarray(ff, dtype=np.float32).reshape(12) + np.asarray(feedback, dtype=np.float32).reshape(12)
        command = self.safety.build_pressure_command(
            motion_norm12=motion_norm,
            gripper_open=gripper_open,
            now_ns=now_ns,
            state_timestamp_ns=state_timestamp_ns,
            reference_timestamp_ns=reference_timestamp_ns,
            current_state12=current_state12,
            pressure_scale=pressure_scale,
        )
        command.debug.update(
            {
                "feedforward_action12": np.asarray(ff, dtype=np.float32).reshape(12).copy(),
                "closed_loop_delta_action12": np.asarray(feedback, dtype=np.float32).reshape(12).copy(),
                "pre_safety_action12": motion_norm.copy(),
            }
        )
        return command
