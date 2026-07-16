from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from soft_vla.motion_control.reference_generator import ReferenceGenerator
from soft_vla.runtime.shared_state import LatestValue, RobotState, UpperAction
from soft_vla.runtime.timing import PeriodicTimer, TimingStats


@dataclass
class UpperLoopReport:
    steps: int
    overruns: int
    timing: dict


class EpisodeUpperLoop10Hz:
    def __init__(
        self,
        *,
        actions: Iterable[UpperAction],
        state_holder: LatestValue,
        reference_holder: LatestValue,
        reference_generator: ReferenceGenerator,
        frequency_hz: float = 10.0,
    ) -> None:
        self.actions = iter(actions)
        self.state_holder = state_holder
        self.reference_holder = reference_holder
        self.reference_generator = reference_generator
        self.frequency_hz = frequency_hz
        self.stop_event = threading.Event()

    def run(self) -> UpperLoopReport:
        timer = PeriodicTimer(self.frequency_hz)
        timing = TimingStats()
        steps = 0
        overruns = 0
        for action in self.actions:
            if self.stop_event.is_set():
                break
            start = time.monotonic_ns()
            state: RobotState | None = self.state_holder.get()
            if state is None:
                state12 = np.zeros(12, dtype=np.float32)
            else:
                state12 = state.state12
            segment = self.reference_generator.build(current_state12=state12, action=action)
            self.reference_holder.set(segment)
            timing.add_ns(time.monotonic_ns() - start)
            if not timer.wait_next():
                overruns += 1
            steps += 1
        return UpperLoopReport(steps=steps, overruns=overruns, timing=timing.summary())

    def stop(self) -> None:
        self.stop_event.set()

