from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


def monotonic_ns() -> int:
    return time.monotonic_ns()


def sleep_until_ns(deadline_ns: int) -> bool:
    now = time.monotonic_ns()
    remaining_ns = deadline_ns - now
    if remaining_ns <= 0:
        return False
    time.sleep(remaining_ns / 1_000_000_000.0)
    return True


@dataclass
class PeriodicTimer:
    frequency_hz: float
    next_deadline_ns: int = field(init=False)
    period_ns: int = field(init=False)

    def __post_init__(self) -> None:
        if self.frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive")
        self.period_ns = int(round(1_000_000_000.0 / float(self.frequency_hz)))
        self.next_deadline_ns = time.monotonic_ns() + self.period_ns

    def wait_next(self) -> bool:
        slept = sleep_until_ns(self.next_deadline_ns)
        self.next_deadline_ns += self.period_ns
        return slept


@dataclass
class TimingStats:
    samples_ms: list[float] = field(default_factory=list)

    def add_ns(self, duration_ns: int) -> None:
        self.samples_ms.append(float(duration_ns) / 1_000_000.0)

    def summary(self) -> dict[str, float | int]:
        if not self.samples_ms:
            return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0}
        arr = np.asarray(self.samples_ms, dtype=np.float64)
        return {
            "count": int(arr.size),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "max_ms": float(arr.max()),
        }

