from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from soft_vla.motion_control.controller_runtime import MotionControlRuntime
from soft_vla.motion_control.feedforward_adapters import (
    AwacFeedforwardAdapter,
    AwacFeedforwardConfig,
    FeedforwardPolicy,
    FeedforwardPressureConfig,
    FeedforwardPressureMLPAdapter,
)
from soft_vla.motion_control.feedback_controllers import (
    IntegralFeedbackConfig,
    IntegralFeedbackController,
    load_fixed_gain,
    make_integral_lqr_q_weights,
    solve_integral_lqr,
)
from soft_vla.motion_control.koopman_adapter import KoopmanAdapter, KoopmanAdapterConfig
from soft_vla.real_robot.safety_manager import SafetyLimits, SafetyManager


FeedforwardBuilder = Callable[["MotionPolicyConfig"], FeedforwardPolicy]
FeedbackBuilder = Callable[["MotionPolicyConfig", KoopmanAdapter], IntegralFeedbackController | None]


@dataclass(frozen=True)
class MotionPolicyConfig:
    feedforward: str = "pressure_model"
    feedback: str = "fixed_k_integral"
    pressure_checkpoint: str | Path = Path(
        "motion_control_training/feedforward_pressure/runs/optimized_state12_raw_pressure/best.pt"
    )
    awac_checkpoint: str | Path = Path(
        "motion_control_training/KORL/runs/feedforward/awac_quadq_2k_eval_2x256/best.pt"
    )
    koopman_checkpoint: str | Path = Path(
        "motion_control_training/koopman/runs/"
        "robot_records_7_03_1_delta_tcp_10hz_to_50hz_k50_epoch1500_wandb_online_20260706_2159/best.pt"
    )
    fixed_k_path: str | Path | None = None
    device: str = "cpu"
    feedback_gain_scale: float = 0.1
    max_integral_error: float = 0.5
    q_tcp6_weight: float = 1.0
    q_state_tail_weight: float = 0.1
    q_latent_weight: float = 0.1
    q_integral_weight: float = 0.5
    r_weight: float = 50.0
    safety_slew_rate_physical_per_s: float | None = 3.0


@dataclass(frozen=True)
class BuiltMotionPolicy:
    runtime: MotionControlRuntime
    koopman: KoopmanAdapter
    metadata: dict


_FEEDFORWARD_BUILDERS: dict[str, FeedforwardBuilder] = {}
_FEEDBACK_BUILDERS: dict[str, FeedbackBuilder] = {}


def register_feedforward(name: str, builder: FeedforwardBuilder) -> None:
    _FEEDFORWARD_BUILDERS[name] = builder


def register_feedback(name: str, builder: FeedbackBuilder) -> None:
    _FEEDBACK_BUILDERS[name] = builder


def build_motion_policy(config: MotionPolicyConfig) -> BuiltMotionPolicy:
    feedforward = _build_feedforward(config)
    koopman = KoopmanAdapter(KoopmanAdapterConfig(checkpoint=config.koopman_checkpoint, device=config.device))
    feedback = _build_feedback(config, koopman)
    runtime = MotionControlRuntime(
        feedforward=feedforward,
        feedback=feedback,
        safety=SafetyManager(SafetyLimits(slew_rate_physical_per_s=config.safety_slew_rate_physical_per_s)),
    )
    return BuiltMotionPolicy(
        runtime=runtime,
        koopman=koopman,
        metadata={
            "feedforward": config.feedforward,
            "feedback": config.feedback,
            "device": config.device,
            "pressure_checkpoint": str(config.pressure_checkpoint),
            "awac_checkpoint": str(config.awac_checkpoint),
            "koopman_checkpoint": str(config.koopman_checkpoint),
            "feedback_gain_scale": config.feedback_gain_scale,
            "max_integral_error": config.max_integral_error,
            "q_tcp6_weight": config.q_tcp6_weight,
            "q_state_tail_weight": config.q_state_tail_weight,
            "q_latent_weight": config.q_latent_weight,
            "q_integral_weight": config.q_integral_weight,
            "r_weight": config.r_weight,
            "fixed_k_source": "path" if config.fixed_k_path is not None else "computed_in_memory",
            "fixed_k_path": None if config.fixed_k_path is None else str(config.fixed_k_path),
        },
    )


def _build_feedforward(config: MotionPolicyConfig) -> FeedforwardPolicy:
    if config.feedforward in _FEEDFORWARD_BUILDERS:
        return _FEEDFORWARD_BUILDERS[config.feedforward](config)
    if config.feedforward == "pressure_model":
        return FeedforwardPressureMLPAdapter(
            FeedforwardPressureConfig(
                checkpoint=config.pressure_checkpoint,
                device=config.device,
                input_mode="target_state",
            )
        )
    if config.feedforward == "awac":
        return AwacFeedforwardAdapter(AwacFeedforwardConfig(checkpoint=config.awac_checkpoint, device=config.device))
    raise ValueError(f"unsupported feedforward policy: {config.feedforward}")


def _build_feedback(config: MotionPolicyConfig, koopman: KoopmanAdapter) -> IntegralFeedbackController | None:
    if config.feedback in _FEEDBACK_BUILDERS:
        return _FEEDBACK_BUILDERS[config.feedback](config, koopman)
    if config.feedback == "none":
        return None
    C = koopman.output_matrix(6)
    if config.feedback == "fixed_k_integral" and config.fixed_k_path is not None:
        K = load_fixed_gain(config.fixed_k_path)
    elif config.feedback in {"fixed_k_integral", "integral_lqr"}:
        q_weights = make_integral_lqr_q_weights(
            n_koopman=koopman.n_koopman,
            ny=6,
            tcp6_weight=config.q_tcp6_weight,
            state_tail_weight=config.q_state_tail_weight,
            latent_weight=config.q_latent_weight,
            integral_weight=config.q_integral_weight,
        )
        K, _, _ = solve_integral_lqr(
            koopman.A_lift,
            koopman.B,
            C,
            q_weights=q_weights,
            r_weight=config.r_weight,
        )
    else:
        raise ValueError(f"unsupported feedback policy: {config.feedback}")
    return IntegralFeedbackController(
        K=np.asarray(K, dtype=np.float64),
        C=C,
        config=IntegralFeedbackConfig(
            ny=6,
            max_integral_error=config.max_integral_error,
            feedback_gain_scale=config.feedback_gain_scale,
        ),
    )
