from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HumanActionMapperConfig:
    joystick_deadzone: float = 0.15
    intervention_release_deadzone: float = 0.10
    max_delta_pos_per_tick: tuple[float, float, float] = (0.001, 0.001, 0.001)
    max_delta_rot_per_tick: tuple[float, float, float] = (0.005, 0.005, 0.005)
    xz_single_axis_lock: bool = True
    translation_xy_enabled: bool = True
    translation_z_enabled: bool = True
    rotation_enabled: bool = False
    rotation_axis: str = "none"
    max_action_slew_pos: float = 0.001
    max_action_slew_rot: float = 0.005
    workspace_min: tuple[float, float, float] | None = None
    workspace_max: tuple[float, float, float] | None = None


@dataclass
class HumanCommand:
    action7: np.ndarray
    active: bool
    input_norm: float
    gripper_command: int | None = None
    success_pressed: bool = False
    failure_pressed: bool = False
    esc_pressed: bool = False
    gamepad_connected: bool = True
    warnings: list[str] | None = None
    raw: dict[str, Any] | None = None


class HumanActionMapper:
    """Map normalized Xbox-style input into a bounded 7D delta TCP action."""

    def __init__(self, config: HumanActionMapperConfig | None = None) -> None:
        self.config = config or HumanActionMapperConfig()
        if self.config.rotation_axis not in {"none", "roll", "pitch", "yaw", "pitch_yaw"}:
            raise ValueError("rotation_axis must be one of none|roll|pitch|yaw|pitch_yaw")
        self.previous_action = np.zeros(7, dtype=np.float32)
        self.previous_action[6] = 1.0
        self._axis_active = False

    def reset(self, *, gripper_open: float = 1.0) -> None:
        self.previous_action = np.zeros(7, dtype=np.float32)
        self.previous_action[6] = 1.0 if gripper_open >= 0.5 else 0.0
        self._axis_active = False

    def sync_gripper_state(self, gripper_open: float) -> None:
        """Synchronize the latched human gripper state with the executed action."""
        self.previous_action[6] = 1.0 if float(gripper_open) >= 0.5 else 0.0

    def map_input(self, state: dict[str, Any], *, current_state12: np.ndarray | None = None) -> HumanCommand:
        cfg = self.config
        lx = _axis_value(state, "left_x")
        ly = _axis_value(state, "left_y")
        rx = _axis_value(state, "right_x")
        ry = _axis_value(state, "right_y")
        lt = _trigger_value(state, "lt")
        rt = _trigger_value(state, "rt")
        buttons = state.get("buttons", {}) or {}
        connected = bool(state.get("connected", True))
        warnings: list[str] = []

        trigger_axis = rt - lt
        dx = _apply_deadzone(ly, cfg.joystick_deadzone) * cfg.max_delta_pos_per_tick[0]
        dy = _apply_deadzone(trigger_axis, cfg.joystick_deadzone) * cfg.max_delta_pos_per_tick[1]
        dz = _apply_deadzone(lx, cfg.joystick_deadzone) * cfg.max_delta_pos_per_tick[2]
        if cfg.xz_single_axis_lock:
            if abs(dx) >= abs(dz):
                dz = 0.0
            else:
                dx = 0.0
        if not cfg.translation_xy_enabled:
            dx = 0.0
            dz = 0.0
        if not cfg.translation_z_enabled:
            dy = 0.0

        droll = dpitch = dyaw = 0.0
        rot_raw = 0.0
        if cfg.rotation_enabled and cfg.rotation_axis != "none":
            if abs(rx) >= abs(ry):
                rot_raw = rx
                rot_input = _apply_deadzone(rx, cfg.joystick_deadzone)
                dominant_axis = "yaw"
            else:
                rot_raw = ry
                rot_input = _apply_deadzone(ry, cfg.joystick_deadzone)
                dominant_axis = "pitch"
            if cfg.rotation_axis == "roll":
                droll = rot_input * cfg.max_delta_rot_per_tick[0]
            elif cfg.rotation_axis == "pitch":
                dpitch = rot_input * cfg.max_delta_rot_per_tick[1]
            elif cfg.rotation_axis == "yaw":
                dyaw = rot_input * cfg.max_delta_rot_per_tick[2]
            elif cfg.rotation_axis == "pitch_yaw":
                if dominant_axis == "pitch":
                    dpitch = rot_input * cfg.max_delta_rot_per_tick[1]
                else:
                    dyaw = rot_input * cfg.max_delta_rot_per_tick[2]

        gripper_command = None
        if bool(buttons.get("a", False)):
            gripper_command = 0
        if bool(buttons.get("y", False)):
            gripper_command = 1
        gripper = self.previous_action[6] if gripper_command is None else float(gripper_command)

        action = np.asarray([dx, dy, dz, droll, dpitch, dyaw, gripper], dtype=np.float32)
        action[:6] = self._apply_workspace_limit(action[:6], current_state12, warnings)
        action[:6] = self._apply_slew_limit(action[:6])

        axis_norm = float(max(abs(lx), abs(ly), abs(trigger_axis), abs(rot_raw) if cfg.rotation_enabled else 0.0))
        axis_threshold = cfg.intervention_release_deadzone if self._axis_active else cfg.joystick_deadzone
        axis_active = bool(connected and axis_norm > axis_threshold)
        active = bool(connected and (axis_active or gripper_command is not None))
        self._axis_active = axis_active
        self.previous_action = action.copy()
        return HumanCommand(
            action7=action,
            active=active,
            input_norm=axis_norm,
            gripper_command=gripper_command,
            success_pressed=bool(buttons.get("x", False)),
            failure_pressed=bool(buttons.get("b", False)),
            esc_pressed=bool(state.get("esc", False)),
            gamepad_connected=connected,
            warnings=warnings,
            raw=state,
        )

    def _apply_slew_limit(self, delta6: np.ndarray) -> np.ndarray:
        prev = self.previous_action[:6]
        out = np.asarray(delta6, dtype=np.float32).copy()
        pos_slew = float(self.config.max_action_slew_pos)
        rot_slew = float(self.config.max_action_slew_rot)
        out[:3] = prev[:3] + np.clip(out[:3] - prev[:3], -pos_slew, pos_slew)
        out[3:6] = prev[3:6] + np.clip(out[3:6] - prev[3:6], -rot_slew, rot_slew)
        return out

    def _apply_workspace_limit(
        self,
        delta6: np.ndarray,
        current_state12: np.ndarray | None,
        warnings: list[str],
    ) -> np.ndarray:
        if current_state12 is None or self.config.workspace_min is None or self.config.workspace_max is None:
            return np.asarray(delta6, dtype=np.float32)
        out = np.asarray(delta6, dtype=np.float32).copy()
        pos = np.asarray(current_state12[:3], dtype=np.float32)
        lo = np.asarray(self.config.workspace_min, dtype=np.float32)
        hi = np.asarray(self.config.workspace_max, dtype=np.float32)
        for idx, axis in enumerate(("x", "y", "z")):
            if pos[idx] <= lo[idx] and out[idx] < 0:
                out[idx] = 0.0
                warnings.append(f"workspace_min_{axis}")
            if pos[idx] >= hi[idx] and out[idx] > 0:
                out[idx] = 0.0
                warnings.append(f"workspace_max_{axis}")
        return out


def _axis_value(state: dict[str, Any], key: str) -> float:
    return float(np.clip(float((state.get("axes", {}) or {}).get(key, 0.0)), -1.0, 1.0))


def _trigger_value(state: dict[str, Any], key: str) -> float:
    raw = float((state.get("axes", {}) or {}).get(key, 0.0))
    return float(np.clip(raw, 0.0, 1.0))


def _apply_deadzone(value: float, deadzone: float) -> float:
    value = float(np.clip(value, -1.0, 1.0))
    deadzone = max(0.0, min(float(deadzone), 0.99))
    if abs(value) <= deadzone:
        return 0.0
    return float(np.sign(value) * ((abs(value) - deadzone) / (1.0 - deadzone)))
