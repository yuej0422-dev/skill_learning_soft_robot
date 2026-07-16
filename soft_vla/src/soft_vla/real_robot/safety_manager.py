from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from soft_vla.motion_control.reference_generator import gripper_open_to_pressure4
from soft_vla.runtime.shared_state import PressureCommand


@dataclass(frozen=True)
class SafetyLimits:
    pressure_norm_min: float = 0.0
    pressure_norm_max: float = 1.0
    physical_pressure_max: float = 3.0
    slew_rate_physical_per_s: float | None = 6.0
    state_timeout_s: float = 0.1
    command_timeout_s: float = 0.25
    workspace_min: tuple[float, float, float] | None = None
    workspace_max: tuple[float, float, float] | None = None


class SafetyManager:
    def __init__(self, limits: SafetyLimits | None = None) -> None:
        self.limits = limits or SafetyLimits()
        self._last_physical12: np.ndarray | None = None
        self._last_command_ns: int | None = None
        self.estop = False

    def reset(self) -> None:
        self._last_physical12 = None
        self._last_command_ns = None
        self.estop = False

    def request_estop(self) -> None:
        self.estop = True

    def build_pressure_command(
        self,
        *,
        motion_norm12: np.ndarray,
        gripper_open: float,
        now_ns: int | None = None,
        state_timestamp_ns: int | None = None,
        reference_timestamp_ns: int | None = None,
        current_state12: np.ndarray | None = None,
        pressure_scale: float = 1.0,
    ) -> PressureCommand:
        now_ns = time.monotonic_ns() if now_ns is None else int(now_ns)
        flags: list[str] = []
        motion = np.asarray(motion_norm12, dtype=np.float32).reshape(-1)
        if motion.shape != (12,):
            raise ValueError(f"motion_norm12 must have shape (12,), got {motion.shape}")
        if not np.all(np.isfinite(motion)):
            flags.append("nan_or_inf_motion_pressure")
            motion = np.nan_to_num(motion, nan=0.0, posinf=0.0, neginf=0.0)

        if self.estop:
            flags.append("estop")
            motion = np.zeros(12, dtype=np.float32)

        if state_timestamp_ns is not None:
            age_s = (now_ns - int(state_timestamp_ns)) / 1_000_000_000.0
            if age_s > self.limits.state_timeout_s:
                flags.append("state_timeout")
                motion = np.zeros(12, dtype=np.float32)
        if reference_timestamp_ns is not None:
            age_s = (now_ns - int(reference_timestamp_ns)) / 1_000_000_000.0
            if age_s > self.limits.command_timeout_s:
                flags.append("command_timeout")
                motion = np.zeros(12, dtype=np.float32)

        if current_state12 is not None and self._outside_workspace(current_state12):
            flags.append("workspace_limit")
            motion = np.zeros(12, dtype=np.float32)

        motion = np.clip(motion * float(pressure_scale), self.limits.pressure_norm_min, self.limits.pressure_norm_max)
        physical12 = np.clip(motion * self.limits.physical_pressure_max, 0.0, self.limits.physical_pressure_max)
        physical12 = self._apply_slew_limit(physical12.astype(np.float32), now_ns, flags)
        gripper4 = gripper_open_to_pressure4(gripper_open)
        final16 = np.concatenate([physical12, gripper4]).astype(np.float32)
        self._last_physical12 = physical12.copy()
        self._last_command_ns = now_ns
        return PressureCommand(
            motion_norm12=physical12 / self.limits.physical_pressure_max,
            motion_physical12=physical12,
            gripper_physical4=gripper4,
            final_physical=final16,
            safety_flags=tuple(flags),
        )

    def _outside_workspace(self, state12: np.ndarray) -> bool:
        xyz = np.asarray(state12, dtype=np.float32).reshape(-1)[:3]
        if self.limits.workspace_min is not None:
            if np.any(xyz < np.asarray(self.limits.workspace_min, dtype=np.float32)):
                return True
        if self.limits.workspace_max is not None:
            if np.any(xyz > np.asarray(self.limits.workspace_max, dtype=np.float32)):
                return True
        return False

    def _apply_slew_limit(self, physical12: np.ndarray, now_ns: int, flags: list[str]) -> np.ndarray:
        if self.limits.slew_rate_physical_per_s is None or self._last_physical12 is None or self._last_command_ns is None:
            return physical12
        dt_s = max((now_ns - self._last_command_ns) / 1_000_000_000.0, 1e-6)
        max_delta = float(self.limits.slew_rate_physical_per_s) * dt_s
        limited = np.clip(physical12, self._last_physical12 - max_delta, self._last_physical12 + max_delta)
        if not np.allclose(limited, physical12):
            flags.append("slew_limited")
        return limited.astype(np.float32)

