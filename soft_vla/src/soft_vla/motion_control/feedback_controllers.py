from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class IntegralFeedbackConfig:
    dt: float = 0.02
    ny: int = 6
    max_integral_error: float = 0.5
    feedback_gain_scale: float = 0.2


class IntegralFeedbackController:
    def __init__(self, K: np.ndarray, C: np.ndarray, config: IntegralFeedbackConfig | None = None) -> None:
        self.config = config or IntegralFeedbackConfig()
        self.K = np.asarray(K, dtype=np.float64)
        self.C = np.asarray(C, dtype=np.float64)
        if self.C.ndim != 2 or self.C.shape[0] != self.config.ny:
            raise ValueError(f"C must have shape [ny,n_lift], got {self.C.shape}")
        expected_cols = self.C.shape[1] + self.config.ny
        if self.K.shape != (12, expected_cols):
            raise ValueError(f"K must have shape {(12, expected_cols)}, got {self.K.shape}")
        self.q = np.zeros((self.config.ny,), dtype=np.float64)

    def reset(self) -> None:
        self.q[:] = 0.0

    def predict(self, lifted_error: np.ndarray) -> np.ndarray:
        e = np.asarray(lifted_error, dtype=np.float64).reshape(-1)
        if e.shape[0] != self.C.shape[1]:
            raise ValueError(f"lifted_error dim {e.shape[0]} does not match C width {self.C.shape[1]}")
        tracking_error = self.C @ e
        self.q = self.q + float(self.config.dt) * tracking_error
        lim = float(self.config.max_integral_error)
        self.q = np.clip(self.q, -lim, lim)
        augmented = np.concatenate([e, self.q])
        feedback = -self.K @ augmented
        return (feedback * float(self.config.feedback_gain_scale)).astype(np.float32)


def make_integral_lqr_q_weights(
    *,
    n_koopman: int,
    ny: int,
    tcp6_weight: float = 1.0,
    state_tail_weight: float = 0.1,
    latent_weight: float = 0.1,
    integral_weight: float = 0.5,
) -> np.ndarray:
    """Build diagonal Q weights for [lifted_error, integral_error].

    The first 12 lifted coordinates are treated as normalized state
    coordinates: TCP pose in dims 0:6 and velocity/tail state in dims 6:12.
    Any remaining lifted coordinates receive ``latent_weight``.
    """

    if n_koopman < 12:
        raise ValueError("n_koopman must be at least 12")
    if ny < 0:
        raise ValueError("ny must be non-negative")
    weights = np.full(n_koopman + ny, float(latent_weight), dtype=np.float64)
    weights[:6] = float(tcp6_weight)
    weights[6:12] = float(state_tail_weight)
    if ny:
        weights[n_koopman:] = float(integral_weight)
    return weights


def build_augmented_system(A_lift: np.ndarray, B: np.ndarray, C: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    A_lift = np.asarray(A_lift, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    ny = C.shape[0]
    At = np.block(
        [
            [A_lift.T, np.zeros((A_lift.shape[0], ny), dtype=np.float64)],
            [float(dt) * C, np.eye(ny, dtype=np.float64)],
        ]
    )
    Bt = np.concatenate([B.T, np.zeros((ny, B.shape[0]), dtype=np.float64)], axis=0)
    return At, Bt


def solve_integral_lqr(
    A_lift: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    *,
    dt: float = 0.02,
    q_weights: np.ndarray | None = None,
    r_weight: float = 10.0,
    max_iterations: int = 10000,
    tolerance: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    At, Bt = build_augmented_system(A_lift, B, C, dt)
    n = At.shape[0]
    if q_weights is None:
        q_weights = np.ones(n, dtype=np.float64)
    Q = np.diag(np.asarray(q_weights, dtype=np.float64).reshape(n))
    R = float(r_weight) * np.eye(Bt.shape[1], dtype=np.float64)
    try:
        import control

        K, _, _ = control.dlqr(At, Bt, Q, R)
        K = np.asarray(K, dtype=np.float64)
    except ImportError:
        try:
            from scipy.linalg import solve_discrete_are
        except ImportError:
            P = solve_discrete_are_iterative(At, Bt, Q, R, max_iterations=max_iterations, tolerance=tolerance)
        else:
            P = solve_discrete_are(At, Bt, Q, R)
        K = np.linalg.solve(R + Bt.T @ P @ Bt, Bt.T @ P @ At)
    return K, At, Bt


def solve_discrete_are_iterative(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    *,
    max_iterations: int = 10000,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """Solve the DARE with fixed-point Riccati iteration using only NumPy.

    This is slower than scipy/control but keeps offline validation runnable in
    minimal environments. It is intended for pre-deployment checks, not online
    control.
    """

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    P = Q.copy()
    for _ in range(int(max_iterations)):
        gain_denom = R + B.T @ P @ B
        P_next = A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(gain_denom, B.T @ P @ A) + Q
        P_next = 0.5 * (P_next + P_next.T)
        if np.linalg.norm(P_next - P, ord="fro") <= float(tolerance) * max(1.0, np.linalg.norm(P, ord="fro")):
            return P_next
        P = P_next
    return P


def save_fixed_gain(path: str | Path, K: np.ndarray, *, At: np.ndarray | None = None, Bt: np.ndarray | None = None) -> None:
    payload = {"K": np.asarray(K, dtype=np.float64)}
    if At is not None:
        payload["At"] = np.asarray(At, dtype=np.float64)
    if Bt is not None:
        payload["Bt"] = np.asarray(Bt, dtype=np.float64)
    np.savez(path, **payload)


def load_fixed_gain(path: str | Path) -> np.ndarray:
    data = np.load(path)
    if "K" not in data:
        raise ValueError(f"fixed gain file {path} does not contain K")
    return np.asarray(data["K"], dtype=np.float64)
