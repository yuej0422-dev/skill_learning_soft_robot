from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soft_vla.runtime.shared_state import ReferenceSegment, UpperAction


@dataclass(frozen=True)
class ReferenceGeneratorConfig:
    upper_frequency_hz: float = 10.0
    control_frequency_hz: float = 50.0
    interpolation: str = "linear"
    delta_tcp_scale: float = 0.2
    max_delta_tcp: tuple[float, float, float, float, float, float] | None = (
        0.02,
        0.02,
        0.02,
        0.08,
        0.08,
        0.08,
    )
    velocity_mode: str = "finite_difference"


class ReferenceGenerator:
    """Expand one 10 Hz delta TCP command into a 50 Hz reference segment."""

    def __init__(self, config: ReferenceGeneratorConfig | None = None) -> None:
        self.config = config or ReferenceGeneratorConfig()
        ratio = self.config.control_frequency_hz / self.config.upper_frequency_hz
        rounded = int(round(ratio))
        if rounded <= 0 or abs(ratio - rounded) > 1e-6:
            raise ValueError(
                "control_frequency_hz must be an integer multiple of upper_frequency_hz; "
                f"got {self.config.control_frequency_hz}/{self.config.upper_frequency_hz}"
            )
        if self.config.interpolation not in {"linear", "zero_order_hold"}:
            raise ValueError(f"unsupported interpolation: {self.config.interpolation}")
        if self.config.velocity_mode not in {"finite_difference", "hold_current", "zero"}:
            raise ValueError(f"unsupported velocity_mode: {self.config.velocity_mode}")
        self.substeps = rounded

    def build(
        self,
        *,
        current_state12: np.ndarray,
        action: UpperAction,
        control_start_step: int | None = None,
    ) -> ReferenceSegment:
        current = np.asarray(current_state12, dtype=np.float32)
        if current.shape != (12,):
            raise ValueError(f"current_state12 must have shape (12,), got {current.shape}")
        if not np.all(np.isfinite(current)):
            raise ValueError("current_state12 contains NaN or Inf")

        scaled_delta = np.asarray(action.delta_tcp6, dtype=np.float32) * float(self.config.delta_tcp_scale)
        if self.config.max_delta_tcp is not None:
            max_delta = np.asarray(self.config.max_delta_tcp, dtype=np.float32)
            if max_delta.shape != (6,):
                raise ValueError("max_delta_tcp must contain 6 values")
            scaled_delta = np.clip(scaled_delta, -max_delta, max_delta)

        refs = np.repeat(current.reshape(1, 12), repeats=self.substeps, axis=0).astype(np.float32)
        if self.config.interpolation == "linear":
            fractions = (np.arange(1, self.substeps + 1, dtype=np.float32) / float(self.substeps)).reshape(-1, 1)
            refs[:, :6] = current[:6] + fractions * scaled_delta.reshape(1, 6)
        else:
            refs[:, :6] = current[:6] + scaled_delta.reshape(1, 6)

        if self.config.velocity_mode == "finite_difference":
            upper_period = 1.0 / float(self.config.upper_frequency_hz)
            refs[:, 6:12] = scaled_delta.reshape(1, 6) / upper_period
        elif self.config.velocity_mode == "zero":
            refs[:, 6:12] = 0.0

        return ReferenceSegment(
            reference_states12=refs,
            gripper_open=action.gripper_open,
            upper_action=action,
            control_start_step=action.upper_step * self.substeps if control_start_step is None else int(control_start_step),
            interpolation=self.config.interpolation,
        )


def gripper_open_to_pressure4(gripper_open: float) -> np.ndarray:
    if float(gripper_open) >= 0.5:
        return np.asarray([3.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return np.asarray([0.0, 3.0, 0.0, 0.0], dtype=np.float32)

