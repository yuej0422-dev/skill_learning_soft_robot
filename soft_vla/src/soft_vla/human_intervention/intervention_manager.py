from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .action_mapper import HumanCommand


@dataclass(frozen=True)
class InterventionManagerConfig:
    release_ticks: int = 1
    handover_blend_steps: int = 2
    blend_tcp_only: bool = True
    blend_gripper: bool = False
    clear_vla_action_queue_on_intervention: bool = False
    request_fresh_vla_on_release: bool = False
    vla_shadow_mode_during_intervention: bool = True
    seamless_policy_resume: bool = True


@dataclass
class ArbitrationResult:
    executed_action7: np.ndarray
    vla_action7: np.ndarray
    human_action7: np.ndarray
    action_source: str
    previous_action_source: str
    intervention_active: bool
    human_input_norm: float
    gamepad_connected: bool
    handover_event: str | None
    handover_blend_active: bool
    handover_blend_step: int
    fallback_used: bool
    reset_triggered: bool = False
    termination_reason: str | None = None


class InterventionManager:
    """Arbitrate VLA shadow actions and human takeover actions at 10 Hz."""

    def __init__(self, config: InterventionManagerConfig | None = None) -> None:
        self.config = config or InterventionManagerConfig()
        self.intervention_active = False
        self.release_counter = 0
        self.previous_source = "vla"
        self.previous_executed = np.zeros(7, dtype=np.float32)
        self.previous_executed[6] = 1.0
        self.blend_remaining = 0
        self.blend_total = max(0, int(self.config.handover_blend_steps))
        self.blend_from = self.previous_executed.copy()

    def step(
        self,
        *,
        vla_action7: np.ndarray | None,
        human_command: HumanCommand | None,
        vla_fallback: bool = False,
    ) -> ArbitrationResult:
        vla = _action_or_hold(vla_action7, self.previous_executed)
        human = _action_or_hold(None if human_command is None else human_command.action7, self.previous_executed)
        # A neutral human gripper input means "do not override".  In
        # particular, do not let HumanActionMapper's old latched value change
        # the gripper merely because TCP intervention started.
        if human_command is not None and human_command.gripper_command is None:
            human[6] = self.previous_executed[6]
        connected = True if human_command is None else bool(human_command.gamepad_connected)
        human_active_now = bool(human_command is not None and human_command.active and connected)
        human_norm = 0.0 if human_command is None else float(human_command.input_norm)
        disconnected_during_intervention = bool(self.intervention_active and human_command is not None and not connected)

        if human_active_now:
            self.intervention_active = True
            self.release_counter = 0
        elif disconnected_during_intervention:
            self.release_counter = 0
        elif self.intervention_active:
            self.release_counter += 1
            if self.release_counter >= max(1, int(self.config.release_ticks)):
                self.intervention_active = False

        if self.intervention_active and connected:
            target = human
            source = "human"
            fallback_used = False
        elif self.intervention_active and not connected:
            target = _hold_action(self.previous_executed)
            source = "fallback"
            fallback_used = True
        elif vla_fallback or vla_action7 is None:
            target = _hold_action(self.previous_executed)
            source = "fallback"
            fallback_used = True
        else:
            target = vla
            source = "vla"
            fallback_used = False

        previous_source = self.previous_source
        handover_event = None
        if source != previous_source:
            handover_event = f"{previous_source}_to_{source}"
            self.blend_from = self.previous_executed.copy()
            self.blend_remaining = self.blend_total if source == "vla" else min(1, self.blend_total)

        executed, blend_active, blend_step = self._smooth(target, source)
        self.previous_executed = executed.copy()
        self.previous_source = source

        termination_reason = None
        reset_triggered = False
        if human_command is not None and human_command.success_pressed:
            termination_reason = "x_success"
            reset_triggered = True
        elif human_command is not None and human_command.failure_pressed:
            termination_reason = "b_failure"
            reset_triggered = True
        elif human_command is not None and human_command.esc_pressed:
            termination_reason = "esc_interrupted"

        return ArbitrationResult(
            executed_action7=executed,
            vla_action7=vla,
            human_action7=human,
            action_source=source,
            previous_action_source=previous_source,
            intervention_active=self.intervention_active,
            human_input_norm=human_norm,
            gamepad_connected=connected,
            handover_event=handover_event,
            handover_blend_active=blend_active,
            handover_blend_step=blend_step,
            fallback_used=fallback_used,
            reset_triggered=reset_triggered,
            termination_reason=termination_reason,
        )

    def _smooth(self, target: np.ndarray, source: str) -> tuple[np.ndarray, bool, int]:
        target = np.asarray(target, dtype=np.float32).copy()
        if self.blend_remaining <= 0:
            return target, False, 0
        total = max(1, self.blend_total)
        step_index = total - self.blend_remaining + 1
        alpha = float(step_index) / float(total)
        out = target.copy()
        out[:6] = (1.0 - alpha) * self.blend_from[:6] + alpha * target[:6]
        if not self.config.blend_gripper:
            out[6] = target[6]
        self.blend_remaining -= 1
        return out.astype(np.float32), True, step_index


def _action_or_hold(action: np.ndarray | None, previous: np.ndarray) -> np.ndarray:
    if action is None:
        return _hold_action(previous)
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.shape != (7,):
        raise ValueError(f"action must have shape (7,), got {arr.shape}")
    arr = arr.copy()
    gripper = float(arr[6])
    if abs(gripper - 0.5) < 1e-6:
        arr[6] = 1.0 if float(np.asarray(previous).reshape(-1)[6]) >= 0.5 else 0.0
    else:
        arr[6] = 1.0 if gripper > 0.5 else 0.0
    return arr


def _hold_action(previous: np.ndarray) -> np.ndarray:
    out = np.zeros(7, dtype=np.float32)
    out[6] = 1.0 if float(np.asarray(previous).reshape(-1)[6]) >= 0.5 else 0.0
    return out
