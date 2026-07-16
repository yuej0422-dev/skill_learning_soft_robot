from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import numpy as np


STATE12_DIM = 12
ACTION7_DIM = 7
MOTION_PRESSURE_DIM = 12
GRIPPER_PRESSURE_DIM = 4
FULL_PRESSURE_DIM = 16


@dataclass(frozen=True)
class RobotState:
    state12: np.ndarray
    gripper_open: float = 0.0
    monotonic_ns: int = 0
    valid: bool = True
    source: str = "unknown"
    debug: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arr = np.asarray(self.state12, dtype=np.float32)
        if arr.shape != (STATE12_DIM,):
            raise ValueError(f"state12 must have shape (12,), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("state12 contains NaN or Inf")
        object.__setattr__(self, "state12", arr)
        object.__setattr__(self, "gripper_open", float(self.gripper_open))

    @property
    def state13(self) -> np.ndarray:
        return np.concatenate([self.state12, np.asarray([self.gripper_open], dtype=np.float32)])


@dataclass(frozen=True)
class UpperAction:
    delta_tcp6: np.ndarray
    gripper_open: float
    upper_step: int
    timestamp: float | None = None
    frame_index: int | None = None
    episode_index: int | None = None
    source: str = "unknown"
    debug: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arr = np.asarray(self.delta_tcp6, dtype=np.float32)
        if arr.shape != (6,):
            raise ValueError(f"delta_tcp6 must have shape (6,), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("delta_tcp6 contains NaN or Inf")
        object.__setattr__(self, "delta_tcp6", arr)
        object.__setattr__(self, "gripper_open", 1.0 if float(self.gripper_open) >= 0.5 else 0.0)
        object.__setattr__(self, "upper_step", int(self.upper_step))

    @property
    def action7(self) -> np.ndarray:
        return np.concatenate([self.delta_tcp6, np.asarray([self.gripper_open], dtype=np.float32)])


@dataclass(frozen=True)
class ReferenceSegment:
    reference_states12: np.ndarray
    gripper_open: float
    upper_action: UpperAction
    control_start_step: int
    interpolation: str
    feedforward_pressures12: np.ndarray | None = None

    def __post_init__(self) -> None:
        refs = np.asarray(self.reference_states12, dtype=np.float32)
        if refs.ndim != 2 or refs.shape[1] != STATE12_DIM:
            raise ValueError(f"reference_states12 must have shape [T,12], got {refs.shape}")
        if not np.all(np.isfinite(refs)):
            raise ValueError("reference_states12 contains NaN or Inf")
        object.__setattr__(self, "reference_states12", refs)
        object.__setattr__(self, "gripper_open", 1.0 if float(self.gripper_open) >= 0.5 else 0.0)
        if self.feedforward_pressures12 is not None:
            ff = np.asarray(self.feedforward_pressures12, dtype=np.float32)
            if ff.shape != (refs.shape[0], MOTION_PRESSURE_DIM):
                raise ValueError(f"feedforward_pressures12 must have shape {(refs.shape[0], 12)}, got {ff.shape}")
            if not np.all(np.isfinite(ff)):
                raise ValueError("feedforward_pressures12 contains NaN or Inf")
            object.__setattr__(self, "feedforward_pressures12", ff)

    def reference_for_substep(self, substep: int) -> np.ndarray:
        idx = min(max(int(substep), 0), self.reference_states12.shape[0] - 1)
        return self.reference_states12[idx]

    def feedforward_for_substep(self, substep: int) -> np.ndarray | None:
        if self.feedforward_pressures12 is None:
            return None
        idx = min(max(int(substep), 0), self.feedforward_pressures12.shape[0] - 1)
        return self.feedforward_pressures12[idx]


@dataclass(frozen=True)
class PressureCommand:
    motion_norm12: np.ndarray
    motion_physical12: np.ndarray
    gripper_physical4: np.ndarray
    final_physical: np.ndarray
    safety_flags: tuple[str, ...] = ()
    debug: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        motion_norm = np.asarray(self.motion_norm12, dtype=np.float32)
        motion_physical = np.asarray(self.motion_physical12, dtype=np.float32)
        gripper = np.asarray(self.gripper_physical4, dtype=np.float32)
        final = np.asarray(self.final_physical, dtype=np.float32)
        if motion_norm.shape != (MOTION_PRESSURE_DIM,):
            raise ValueError(f"motion_norm12 must have shape (12,), got {motion_norm.shape}")
        if motion_physical.shape != (MOTION_PRESSURE_DIM,):
            raise ValueError(f"motion_physical12 must have shape (12,), got {motion_physical.shape}")
        if gripper.shape != (GRIPPER_PRESSURE_DIM,):
            raise ValueError(f"gripper_physical4 must have shape (4,), got {gripper.shape}")
        if final.shape != (FULL_PRESSURE_DIM,):
            raise ValueError(f"final_physical must have shape (16,), got {final.shape}")
        for name, arr in [
            ("motion_norm12", motion_norm),
            ("motion_physical12", motion_physical),
            ("gripper_physical4", gripper),
            ("final_physical", final),
        ]:
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains NaN or Inf")
        object.__setattr__(self, "motion_norm12", motion_norm)
        object.__setattr__(self, "motion_physical12", motion_physical)
        object.__setattr__(self, "gripper_physical4", gripper)
        object.__setattr__(self, "final_physical", final)


class LatestValue:
    """Small lock-protected holder for cross-thread handoff."""

    def __init__(self, initial=None) -> None:
        self._value = initial
        self._lock = Lock()

    def set(self, value) -> None:
        with self._lock:
            self._value = value

    def get(self):
        with self._lock:
            return self._value

