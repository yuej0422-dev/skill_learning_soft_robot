from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HumanTargetIntegratorConfig:
    enabled: bool = True
    max_pos_offset: float = 0.20
    max_rot_offset: float = 1.00
    eps: float = 1e-8


@dataclass
class HumanTargetIntegratorResult:
    action7: np.ndarray
    accumulated_delta6: np.ndarray
    xz_direction: str | None
    y_direction: str | None
    rot_direction: str | None
    reset: bool


class HumanTargetIntegrator:
    """Convert instantaneous manual jog commands into bounded target offsets."""

    def __init__(self, config: HumanTargetIntegratorConfig | None = None) -> None:
        self.config = config or HumanTargetIntegratorConfig()
        self.accumulated_delta6 = np.zeros(6, dtype=np.float32)
        self.last_xz_direction: str | None = None
        self.last_y_direction: str | None = None
        self.last_rot_direction: str | None = None

    def reset(self) -> None:
        self.accumulated_delta6[:] = 0.0
        self.last_xz_direction = None
        self.last_y_direction = None
        self.last_rot_direction = None

    def step(self, action7: np.ndarray, *, active: bool) -> HumanTargetIntegratorResult:
        action = np.asarray(action7, dtype=np.float32).reshape(7).copy()
        if not self.config.enabled or not active:
            self.reset()
            return HumanTargetIntegratorResult(
                action7=action,
                accumulated_delta6=self.accumulated_delta6.copy(),
                xz_direction=None,
                y_direction=None,
                rot_direction=None,
                reset=True,
            )

        delta6 = action[:6].copy()
        xz_direction, xz_axis = _dominant_xz_direction(delta6, self.config.eps)
        y_direction = _axis_direction(delta6[1], "y", self.config.eps)
        rot_direction, rot_axis = _dominant_rot_direction(delta6, self.config.eps)

        self._integrate_group(indices=(0, 2), active_index=xz_axis, direction=xz_direction, last_attr="last_xz_direction", delta6=delta6)
        self._integrate_group(indices=(1,), active_index=1 if y_direction is not None else None, direction=y_direction, last_attr="last_y_direction", delta6=delta6)
        self._integrate_group(indices=(3, 4, 5), active_index=rot_axis, direction=rot_direction, last_attr="last_rot_direction", delta6=delta6)

        pos_limit = abs(float(self.config.max_pos_offset))
        rot_limit = abs(float(self.config.max_rot_offset))
        self.accumulated_delta6[:3] = np.clip(self.accumulated_delta6[:3], -pos_limit, pos_limit)
        self.accumulated_delta6[3:6] = np.clip(self.accumulated_delta6[3:6], -rot_limit, rot_limit)

        out = action.copy()
        out[:6] = self.accumulated_delta6
        return HumanTargetIntegratorResult(
            action7=out,
            accumulated_delta6=self.accumulated_delta6.copy(),
            xz_direction=xz_direction,
            y_direction=y_direction,
            rot_direction=rot_direction,
            reset=False,
        )

    def _integrate_group(
        self,
        *,
        indices: tuple[int, ...],
        active_index: int | None,
        direction: str | None,
        last_attr: str,
        delta6: np.ndarray,
    ) -> None:
        last_direction = getattr(self, last_attr)
        if direction is None or active_index is None:
            for idx in indices:
                self.accumulated_delta6[idx] = 0.0
            setattr(self, last_attr, None)
            return

        if last_direction != direction:
            for idx in indices:
                self.accumulated_delta6[idx] = 0.0
            self.accumulated_delta6[active_index] = delta6[active_index]
        else:
            self.accumulated_delta6[active_index] += delta6[active_index]
            for idx in indices:
                if idx != active_index:
                    self.accumulated_delta6[idx] = 0.0
        setattr(self, last_attr, direction)


def _axis_direction(value: float, name: str, eps: float) -> str | None:
    value = float(value)
    if abs(value) <= eps:
        return None
    return f"{name}+" if value > 0.0 else f"{name}-"


def _dominant_xz_direction(delta6: np.ndarray, eps: float) -> tuple[str | None, int | None]:
    dx = float(delta6[0])
    dz = float(delta6[2])
    if abs(dx) <= eps and abs(dz) <= eps:
        return None, None
    if abs(dx) >= abs(dz):
        return ("x+" if dx > 0.0 else "x-"), 0
    return ("z+" if dz > 0.0 else "z-"), 2


def _dominant_rot_direction(delta6: np.ndarray, eps: float) -> tuple[str | None, int | None]:
    values = [(3, float(delta6[3]), "roll"), (4, float(delta6[4]), "pitch"), (5, float(delta6[5]), "yaw")]
    idx, value, name = max(values, key=lambda item: abs(item[1]))
    if abs(value) <= eps:
        return None, None
    return (f"{name}+" if value > 0.0 else f"{name}-"), idx
