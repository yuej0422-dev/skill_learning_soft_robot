from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from soft_vla.motion_control.controller_runtime import MotionControlRuntime
from soft_vla.motion_control.feedback_controllers import (
    IntegralFeedbackConfig,
    IntegralFeedbackController,
    load_fixed_gain,
    make_integral_lqr_q_weights,
    solve_integral_lqr,
)
from soft_vla.motion_control.feedforward_adapters import FeedforwardPressureConfig, FeedforwardPressureMLPAdapter, ZeroFeedforwardPolicy
from soft_vla.motion_control.koopman_adapter import KoopmanAdapter, KoopmanAdapterConfig
from soft_vla.motion_control.reference_generator import ReferenceGenerator, ReferenceGeneratorConfig
from soft_vla.real_robot.pressure_driver import MockPressureDriver, SerialPressureDriver, SerialPressureDriverConfig
from soft_vla.real_robot.robot_io import LuMoStateSource, LuMoStateSourceConfig, MockRobotStateSource
from soft_vla.real_robot.safety_manager import SafetyLimits, SafetyManager
from soft_vla.real_robot.single_point_plot import save_single_point_plot
from soft_vla.runtime.async_logger import AsyncJsonlLogger
from soft_vla.runtime.shared_state import UpperAction
from soft_vla.runtime.timing import PeriodicTimer, TimingStats


def parse_delta(spec: str) -> np.ndarray:
    values = [float(item) for item in spec.split(",") if item.strip()]
    if len(values) != 6:
        raise argparse.ArgumentTypeError("--target-delta must contain 6 comma-separated values")
    return np.asarray(values, dtype=np.float32)


def build_feedback(
    kind: str,
    fixed_k_path: Path | None,
    *,
    koopman: KoopmanAdapter | None,
    feedback_gain_scale: float,
    max_integral_error: float,
    q_tcp6_weight: float,
    q_state_tail_weight: float,
    q_latent_weight: float,
    q_integral_weight: float,
    r_weight: float,
) -> IntegralFeedbackController | None:
    if kind == "none":
        return None
    if koopman is None:
        raise ValueError("koopman adapter is required when feedback is enabled")
    C = koopman.output_matrix(6)
    if kind == "fixed_k_integral":
        if fixed_k_path is None:
            raise ValueError("--fixed-k-path is required for fixed_k_integral")
        K = load_fixed_gain(fixed_k_path)
    else:
        q_weights = make_integral_lqr_q_weights(
            n_koopman=koopman.n_koopman,
            ny=6,
            tcp6_weight=q_tcp6_weight,
            state_tail_weight=q_state_tail_weight,
            latent_weight=q_latent_weight,
            integral_weight=q_integral_weight,
        )
        K, _, _ = solve_integral_lqr(koopman.A_lift, koopman.B, C, q_weights=q_weights, r_weight=r_weight)
    return IntegralFeedbackController(
        K=K,
        C=C,
        config=IntegralFeedbackConfig(
            ny=6,
            max_integral_error=max_integral_error,
            feedback_gain_scale=feedback_gain_scale,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-amplitude single-point target debug scaffold.")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--hardware-enabled", action="store_true")
    parser.add_argument("--ip", default="192.168.140.1")
    parser.add_argument("--rigid-body-id", type=int, default=1)
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--packet-channels", type=int, choices=[12, 16], default=16)
    parser.add_argument("--target-delta", type=parse_delta, required=True)
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument("--frequency", type=float, default=50.0)
    parser.add_argument("--feedforward", choices=["zero", "pressure_model"], default="zero")
    parser.add_argument("--feedback", choices=["none", "integral_lqr", "fixed_k_integral"], default="none")
    parser.add_argument("--pressure-checkpoint", type=Path, default=Path("motion_control_training/feedforward_pressure/runs/optimized_state12_raw_pressure/best.pt"))
    parser.add_argument(
        "--koopman-checkpoint",
        type=Path,
        default=Path("motion_control_training/koopman/runs/robot_records_7_03_1_delta_tcp_10hz_to_50hz_k50_epoch1500_wandb_online_20260706_2159/best.pt"),
    )
    parser.add_argument("--fixed-k-path", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--delta-tcp-scale", type=float, default=1.0)
    parser.add_argument("--pressure-scale", type=float, default=0.2)
    parser.add_argument("--feedback-gain-scale", type=float, default=0.05)
    parser.add_argument("--max-integral-error", type=float, default=0.5)
    parser.add_argument("--q-tcp6-weight", type=float, default=1.0)
    parser.add_argument("--q-state-tail-weight", type=float, default=0.1)
    parser.add_argument("--q-latent-weight", type=float, default=0.1)
    parser.add_argument("--q-integral-weight", type=float, default=0.5)
    parser.add_argument("--r-weight", type=float, default=10.0)
    parser.add_argument("--log-jsonl", type=Path, default=None)
    parser.add_argument("--plot-path", type=Path, default=None)
    args = parser.parse_args()

    if not args.mock and not args.hardware_enabled:
        raise SystemExit("Use --mock for offline scaffold or --hardware-enabled for real hardware.")
    if args.pressure_scale < 0 or args.pressure_scale > 1.0:
        raise SystemExit("--pressure-scale must be in [0, 1] for single-point debug.")
    if args.mock:
        state_source = MockRobotStateSource(state12=np.zeros(12, dtype=np.float32), gripper_open=1.0)
        driver = MockPressureDriver(packet_channels=args.packet_channels)
    else:
        state_source = LuMoStateSource(LuMoStateSourceConfig(ip=args.ip, rigid_body_id=args.rigid_body_id))
        driver = SerialPressureDriver(
            SerialPressureDriverConfig(port=args.port, baudrate=args.baudrate, packet_channels=args.packet_channels)
        )

    if args.feedforward == "pressure_model":
        feedforward = FeedforwardPressureMLPAdapter(
            FeedforwardPressureConfig(checkpoint=args.pressure_checkpoint, device=args.device, input_mode="target_state")
        )
    else:
        feedforward = ZeroFeedforwardPolicy()
    koopman = None
    if args.feedback != "none":
        koopman = KoopmanAdapter(KoopmanAdapterConfig(checkpoint=args.koopman_checkpoint, device=args.device))
    feedback = build_feedback(
        args.feedback,
        args.fixed_k_path,
        koopman=koopman,
        feedback_gain_scale=args.feedback_gain_scale,
        max_integral_error=args.max_integral_error,
        q_tcp6_weight=args.q_tcp6_weight,
        q_state_tail_weight=args.q_state_tail_weight,
        q_latent_weight=args.q_latent_weight,
        q_integral_weight=args.q_integral_weight,
        r_weight=args.r_weight,
    )
    runtime = MotionControlRuntime(
        feedforward=feedforward,
        feedback=feedback,
        safety=SafetyManager(SafetyLimits(slew_rate_physical_per_s=3.0)),
    )
    ref_gen = ReferenceGenerator(ReferenceGeneratorConfig(delta_tcp_scale=args.delta_tcp_scale))
    if args.log_jsonl is not None and args.log_jsonl.exists():
        args.log_jsonl.unlink()
    logger = AsyncJsonlLogger(args.log_jsonl) if args.log_jsonl else None
    timing = TimingStats()
    state_source.open()
    driver.open()
    if logger:
        logger.start()
    try:
        initial_state = state_source.read_state(blocking=True)
        action = UpperAction(delta_tcp6=args.target_delta, gripper_open=initial_state.gripper_open, upper_step=0, source="single_point")
        segment = ref_gen.build(current_state12=initial_state.state12, action=action)
        steps = max(1, int(round(args.duration_s * args.frequency)))
        timer = PeriodicTimer(args.frequency)
        for step in range(steps):
            t0 = time.monotonic_ns()
            current = state_source.read_state(blocking=True)
            substep = min(step, segment.reference_states12.shape[0] - 1)
            ref = segment.reference_for_substep(substep)
            lifted_error = None if feedback is None else koopman.tracking_error(current.state12, ref)
            cmd = runtime.compute(
                current_state12=current.state12,
                reference_state12=ref,
                delta_tcp6=action.delta_tcp6,
                gripper_open=action.gripper_open,
                lifted_error=lifted_error,
                pressure_scale=args.pressure_scale,
                now_ns=t0,
                state_timestamp_ns=current.monotonic_ns,
                reference_timestamp_ns=t0,
            )
            written = driver.send_physical(cmd.final_physical)
            timing.add_ns(time.monotonic_ns() - t0)
            if logger:
                logger.log(
                    {
                        "step": step,
                        "substep": substep,
                        "state": current.state13.tolist(),
                        "reference": ref.tolist(),
                        "motion_norm12": cmd.motion_norm12.tolist(),
                        "feedforward_action12": cmd.debug["feedforward_action12"].tolist(),
                        "closed_loop_delta_action12": cmd.debug["closed_loop_delta_action12"].tolist(),
                        "pre_safety_action12": cmd.debug["pre_safety_action12"].tolist(),
                        "motion_physical12": cmd.motion_physical12.tolist(),
                        "gripper_physical4": cmd.gripper_physical4.tolist(),
                        "pressure": cmd.final_physical.tolist(),
                        "integral_state": [] if feedback is None else feedback.q.tolist(),
                        "flags": list(cmd.safety_flags),
                        "written": written,
                    }
                )
            timer.wait_next()
    finally:
        try:
            driver.send_zero()
        finally:
            driver.close()
            state_source.close()
            if logger:
                logger.close()

    plot_path = None
    if args.plot_path is not None:
        if args.log_jsonl is None:
            raise SystemExit("--plot-path requires --log-jsonl")
        plot_path = save_single_point_plot(args.log_jsonl, args.plot_path, frequency=args.frequency)

    print(
        json.dumps(
            {
                "mode": "mock" if args.mock else "hardware",
                "feedforward": args.feedforward,
                "feedback": args.feedback,
                "pressure_scale": args.pressure_scale,
                "plot_path": None if plot_path is None else str(plot_path),
                "timing": timing.summary(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
