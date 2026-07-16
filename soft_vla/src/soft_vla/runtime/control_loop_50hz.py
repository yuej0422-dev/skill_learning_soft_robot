from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from soft_vla.motion_control.controller_runtime import MotionControlRuntime
from soft_vla.real_robot.pressure_driver import PressureDriver
from soft_vla.real_robot.robot_io import RobotStateSource
from soft_vla.runtime.async_logger import AsyncJsonlLogger
from soft_vla.runtime.shared_state import LatestValue, ReferenceSegment
from soft_vla.runtime.timing import PeriodicTimer, TimingStats


@dataclass
class ControlLoopReport:
    steps: int = 0
    overruns: int = 0
    exceptions: list[str] = field(default_factory=list)
    timing: TimingStats = field(default_factory=TimingStats)


class ControlLoop50Hz:
    def __init__(
        self,
        *,
        state_source: RobotStateSource,
        pressure_driver: PressureDriver,
        controller: MotionControlRuntime,
        reference_holder: LatestValue,
        frequency_hz: float = 50.0,
        logger: AsyncJsonlLogger | None = None,
        max_steps: int | None = None,
        lifted_error_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.state_source = state_source
        self.pressure_driver = pressure_driver
        self.controller = controller
        self.reference_holder = reference_holder
        self.frequency_hz = frequency_hz
        self.logger = logger
        self.max_steps = max_steps
        self.lifted_error_fn = lifted_error_fn
        self.stop_event = threading.Event()
        self.report = ControlLoopReport()

    def run(self) -> ControlLoopReport:
        self.state_source.open()
        self.pressure_driver.open()
        if self.logger is not None:
            self.logger.start()
        timer = PeriodicTimer(self.frequency_hz)
        try:
            while not self.stop_event.is_set():
                if self.max_steps is not None and self.report.steps >= self.max_steps:
                    break
                start_ns = time.monotonic_ns()
                self._step(start_ns)
                duration_ns = time.monotonic_ns() - start_ns
                self.report.timing.add_ns(duration_ns)
                slept = timer.wait_next()
                if not slept:
                    self.report.overruns += 1
                self.report.steps += 1
        finally:
            try:
                self.pressure_driver.send_zero()
            finally:
                self.pressure_driver.close()
                self.state_source.close()
                if self.logger is not None:
                    self.logger.close()
        return self.report

    def stop(self) -> None:
        self.stop_event.set()

    def _step(self, now_ns: int) -> None:
        state = self.state_source.read_state(blocking=True)
        segment: ReferenceSegment | None = self.reference_holder.get()
        if segment is None:
            pressure = self.controller.safety.build_pressure_command(
                motion_norm12=np.zeros(12, dtype=np.float32),
                gripper_open=state.gripper_open,
                now_ns=now_ns,
                state_timestamp_ns=state.monotonic_ns,
                current_state12=state.state12,
            )
            substep = None
        else:
            substep = max(0, self.report.steps - segment.control_start_step)
            ref = segment.reference_for_substep(substep)
            lifted_error = self._lifted_error(state.state12, ref)
            ff = segment.feedforward_for_substep(substep)
            if ff is not None and self.controller.feedforward is not None:
                # Keep real-time code simple: precomputed feedforward is used through a tiny closure-like object.
                class _Precomputed:
                    def predict(self, **kwargs):
                        return ff

                old_ff = self.controller.feedforward
                self.controller.feedforward = _Precomputed()
                try:
                    pressure = self.controller.compute(
                        current_state12=state.state12,
                        reference_state12=ref,
                        delta_tcp6=segment.upper_action.delta_tcp6,
                        gripper_open=segment.gripper_open,
                        lifted_error=lifted_error,
                        now_ns=now_ns,
                        state_timestamp_ns=state.monotonic_ns,
                        reference_timestamp_ns=now_ns,
                    )
                finally:
                    self.controller.feedforward = old_ff
            else:
                pressure = self.controller.compute(
                    current_state12=state.state12,
                    reference_state12=ref,
                    delta_tcp6=segment.upper_action.delta_tcp6,
                    gripper_open=segment.gripper_open,
                    lifted_error=lifted_error,
                    now_ns=now_ns,
                    state_timestamp_ns=state.monotonic_ns,
                    reference_timestamp_ns=now_ns,
            )
        self.pressure_driver.send_physical(pressure.final_physical)
        if self.logger is not None:
            self.logger.log(
                {
                    "loop": "control_50hz",
                    "step": self.report.steps,
                    "substep": substep,
                    "state": state.state13.tolist(),
                    "pressure": pressure.final_physical.tolist(),
                    "flags": list(pressure.safety_flags),
                    "time_ns": now_ns,
                }
            )

    def _lifted_error(self, current_state12: np.ndarray, reference_state12: np.ndarray) -> np.ndarray | None:
        if self.controller.feedback is None:
            return None
        if self.lifted_error_fn is None:
            raise RuntimeError("lifted_error_fn is required when feedback is enabled in the 50 Hz loop")
        return self.lifted_error_fn(current_state12, reference_state12)
