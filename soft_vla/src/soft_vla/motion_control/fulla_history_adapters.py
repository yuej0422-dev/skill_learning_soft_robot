from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PhysicalBarPressureConfig:
    checkpoint: str | Path
    device: str = "cpu"
    physical_pressure_max: float = 3.0


class PhysicalBarPressureMLPAdapter:
    """Adapt a raw-pressure MLP to the runtime's normalized pressure contract.

    The feedforward model is trained directly against ``raw_pressure`` in bar,
    while ``MotionControlRuntime`` expects a value in [0, 1] before converting
    it back to physical pressure.  This adapter performs that unit conversion
    explicitly instead of treating bar values as normalized values.
    """

    def __init__(self, config: PhysicalBarPressureConfig) -> None:
        if float(config.physical_pressure_max) <= 0:
            raise ValueError("physical_pressure_max must be positive")
        self.config = config
        self.policy = _load_pressure_mlp(config.checkpoint, device=config.device)

    def predict(
        self,
        *,
        current_state12: np.ndarray,
        reference_state12: np.ndarray,
        delta_tcp6: np.ndarray,
    ) -> np.ndarray:
        del current_state12, delta_tcp6
        physical_bar = np.asarray(self.policy.predict_pressure(reference_state12), dtype=np.float32).reshape(12)
        physical_bar = np.clip(physical_bar, 0.0, float(self.config.physical_pressure_max))
        return (physical_bar / float(self.config.physical_pressure_max)).astype(np.float32)


class PhysicalBarFeedbackAdapter:
    """Convert a model-space feedback command in bar to normalized pressure."""

    def __init__(self, controller, *, physical_pressure_max: float = 3.0) -> None:
        if float(physical_pressure_max) <= 0:
            raise ValueError("physical_pressure_max must be positive")
        self.controller = controller
        self.physical_pressure_max = float(physical_pressure_max)

    def reset(self) -> None:
        self.controller.reset()

    def predict(self, lifted_error: np.ndarray) -> np.ndarray:
        physical_bar = np.asarray(self.controller.predict(lifted_error), dtype=np.float32)
        return (physical_bar / self.physical_pressure_max).astype(np.float32)


@dataclass(frozen=True)
class FullAHistoryKoopmanConfig:
    checkpoint: str | Path
    device: str = "cpu"
    initial_pressure_bar: float = 0.0


class FullAHistoryKoopmanAdapter:
    """Online adapter for the Full-A history-context Koopman checkpoint.

    Training context at control step t is
    ``[x[t-h+1:t+1], u[t-h:t]]``.  The adapter maintains separate measured and
    reference state histories and a shared history of the pressures that were
    actually sent to the robot.  Sharing the action history makes the lifted
    error represent the state/reference difference under identical past input.
    """

    def __init__(self, config: FullAHistoryKoopmanConfig) -> None:
        self.config = config
        self._load()
        self.reset()

    def _load(self) -> None:
        try:
            import torch
            from motion_control_training.koopman.experiments.fullA_history_v2.model_fullA_history import (
                FullAHistoryKoopmanNetwork,
            )
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("torch and the fullA_history_v2 model are required") from exc

        checkpoint = torch.load(self.config.checkpoint, map_location=self.config.device, weights_only=False)
        metadata = checkpoint.get("metadata", {})
        if metadata.get("experiment") != "fullA_history_v2":
            raise ValueError(f"checkpoint is not fullA_history_v2: {self.config.checkpoint}")

        self.model = FullAHistoryKoopmanNetwork(
            context_dim=int(checkpoint["context_dim"]),
            n_state=int(checkpoint["n_state"]),
            u_dim=int(checkpoint["u_dim"]),
            encode_dim=int(checkpoint["encode_dim"]),
            hidden_sizes=[int(v) for v in checkpoint["hidden_sizes"]],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.to(self.config.device)
        self.model.eval()
        self.torch = torch
        self.metadata = metadata
        self.n_state = int(checkpoint["n_state"])
        self.u_dim = int(checkpoint["u_dim"])
        self.n_koopman = int(checkpoint["n_koopman"])
        self.history_steps = int(metadata["history_steps"])
        self.target_hz = float(metadata["target_hz"])
        self.state_indices = [int(v) for v in metadata.get("state_indices", range(self.n_state))]
        self.pressure_indices = [int(v) for v in metadata.get("pressure_indices", range(self.u_dim))]
        if self.n_state != 12 or self.state_indices != list(range(12)):
            raise ValueError("deployment currently requires a 12-D state checkpoint with state_indices 0:12")
        if self.u_dim != 12 or self.pressure_indices != list(range(12)):
            raise ValueError("deployment currently requires a 12-D pressure checkpoint with pressure_indices 0:12")
        expected_context_dim = self.history_steps * (self.n_state + self.u_dim)
        if int(checkpoint["context_dim"]) != expected_context_dim:
            raise ValueError(
                f"context_dim {checkpoint['context_dim']} does not match history layout {expected_context_dim}"
            )
        self.state_mean = np.asarray(metadata["state_mean"], dtype=np.float32).reshape(self.n_state)
        self.state_std = np.maximum(
            np.asarray(metadata["state_std"], dtype=np.float32).reshape(self.n_state), 1e-6
        )
        # Keep the row-vector convention used by training.  The LQR helper
        # transposes these matrices when constructing its column-vector system.
        self.A_lift = self.model.A.detach().cpu().numpy().astype(np.float64)
        self.B = self.model.B.detach().cpu().numpy().astype(np.float64)

    def reset(
        self,
        current_state12: np.ndarray | None = None,
        reference_state12: np.ndarray | None = None,
    ) -> None:
        self._measured_history: list[np.ndarray] = []
        self._reference_history: list[np.ndarray] = []
        initial_pressure = np.full((self.u_dim,), float(self.config.initial_pressure_bar), dtype=np.float32)
        self._pressure_history: list[np.ndarray] = [initial_pressure.copy() for _ in range(self.history_steps)]
        self._awaiting_control = False
        if current_state12 is not None:
            current = self._state12(current_state12)
            reference = current if reference_state12 is None else self._state12(reference_state12)
            self._measured_history = [current.copy() for _ in range(self.history_steps)]
            self._reference_history = [reference.copy() for _ in range(self.history_steps)]

    def normalize_state(self, state12: np.ndarray) -> np.ndarray:
        return ((self._state12(state12) - self.state_mean) / self.state_std).astype(np.float32)

    def tracking_error(self, current_state12: np.ndarray, reference_state12: np.ndarray) -> np.ndarray:
        if self._awaiting_control:
            raise RuntimeError("record_control() must be called once after each tracking_error()")
        current = self._state12(current_state12)
        reference = self._state12(reference_state12)
        if not self._measured_history:
            self.reset(current, reference)
        else:
            self._append_bounded(self._measured_history, current)
            self._append_bounded(self._reference_history, reference)

        measured_lift = self._lift_from_history(self._measured_history, current)
        reference_lift = self._lift_from_history(self._reference_history, reference)
        self._awaiting_control = True
        return measured_lift - reference_lift

    def record_control(self, physical_pressure12: np.ndarray) -> None:
        if not self._awaiting_control:
            raise RuntimeError("tracking_error() must be called before record_control()")
        pressure = np.asarray(physical_pressure12, dtype=np.float32).reshape(self.u_dim)
        if not np.all(np.isfinite(pressure)):
            raise ValueError("physical pressure history contains NaN or Inf")
        self._append_bounded(self._pressure_history, pressure)
        self._awaiting_control = False

    def output_matrix(self, ny: int) -> np.ndarray:
        if ny < 0 or ny > self.n_state:
            raise ValueError(f"ny must be between 0 and {self.n_state}")
        c_full = np.concatenate(
            [np.eye(self.n_state), np.zeros((self.n_state, self.n_koopman - self.n_state))], axis=1
        )
        return c_full[:ny].astype(np.float64)

    def history_snapshot(self) -> dict[str, np.ndarray]:
        """Return copies of the online buffers for deployment diagnostics."""
        return {
            "measured_state": np.stack(self._measured_history) if self._measured_history else np.empty((0, 12)),
            "reference_state": np.stack(self._reference_history) if self._reference_history else np.empty((0, 12)),
            "physical_pressure": np.stack(self._pressure_history),
        }

    def _lift_from_history(self, state_history: list[np.ndarray], current_state: np.ndarray) -> np.ndarray:
        normalized_states = np.stack([self.normalize_state(value) for value in state_history], axis=0)
        pressures = np.stack(self._pressure_history, axis=0).astype(np.float32)
        context = np.concatenate([normalized_states.reshape(-1), pressures.reshape(-1)]).astype(np.float32)
        normalized_current = self.normalize_state(current_state)
        with self.torch.no_grad():
            state_tensor = self.torch.as_tensor(
                normalized_current, dtype=self.torch.float32, device=self.config.device
            ).unsqueeze(0)
            context_tensor = self.torch.as_tensor(
                context, dtype=self.torch.float32, device=self.config.device
            ).unsqueeze(0)
            lifted = self.model.encode(state_tensor, context_tensor).detach().cpu().numpy()[0]
        return lifted.astype(np.float64)

    def _append_bounded(self, values: list[np.ndarray], value: np.ndarray) -> None:
        values.append(value.copy())
        if len(values) > self.history_steps:
            del values[0]

    @staticmethod
    def _state12(value: np.ndarray) -> np.ndarray:
        state = np.asarray(value, dtype=np.float32).reshape(12)
        if not np.all(np.isfinite(state)):
            raise ValueError("state history contains NaN or Inf")
        return state


def _load_pressure_mlp(checkpoint: str | Path, *, device: str):
    try:
        from motion_control_training.feedforward_pressure.infer_pressure import load_policy
    except ImportError as exc:  # pragma: no cover - depends on repository environment
        raise RuntimeError("cannot import feedforward pressure loader") from exc
    return load_policy(checkpoint, device=device)
