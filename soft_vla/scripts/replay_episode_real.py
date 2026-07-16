from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from soft_vla.motion_control.controller_runtime import MotionControlRuntime
from soft_vla.motion_control.feedforward_adapters import (
    AwacFeedforwardAdapter,
    AwacFeedforwardConfig,
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
from soft_vla.motion_control.reference_generator import ReferenceGenerator, ReferenceGeneratorConfig
from soft_vla.real_robot.pressure_driver import MockPressureDriver, SerialPressureDriver, SerialPressureDriverConfig
from soft_vla.real_robot.robot_io import LuMoStateSource, LuMoStateSourceConfig, MockRobotStateSource
from soft_vla.real_robot.safety_manager import SafetyLimits, SafetyManager
from soft_vla.real_robot.single_point_plot import save_single_point_plot
from soft_vla.runtime.async_logger import AsyncJsonlLogger
from soft_vla.runtime.shared_state import UpperAction
from soft_vla.runtime.timing import PeriodicTimer, TimingStats


def load_episode(root: Path, episode_index: int) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required. Use the soft_vla_cuda conda env.") from exc
    rows: list[dict] = []
    for path in sorted((root / "data").glob("chunk-*/file-*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "frame_index", "timestamp", "observation.state", "action"])
        data = table.to_pydict()
        for i, ep in enumerate(data["episode_index"]):
            if int(ep) != episode_index:
                continue
            rows.append(
                {
                    "frame_index": int(data["frame_index"][i]),
                    "timestamp": float(data["timestamp"][i]),
                    "state": np.asarray(data["observation.state"][i], dtype=np.float32),
                    "action": np.asarray(data["action"][i], dtype=np.float32),
                }
            )
    rows.sort(key=lambda item: item["frame_index"])
    return rows


def build_feedback(
    kind: str,
    fixed_k_path: Path | None,
    *,
    koopman: KoopmanAdapter,
    feedback_gain_scale: float,
    max_integral_error: float,
    q_tcp6_weight: float,
    q_state_tail_weight: float,
    q_latent_weight: float,
    q_integral_weight: float,
    r_weight: float,
) -> IntegralFeedbackController:
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
    parser = argparse.ArgumentParser(
        description=(
            "Replay a small LeRobot episode segment. Dataset state/action builds the target; "
            "LuMo measured state closes the motion-control loop."
        )
    )
    parser.add_argument("--mock", action="store_true", help="Run without hardware using a perfect-tracking mock state source.")
    parser.add_argument("--hardware-enabled", action="store_true")
    parser.add_argument("--dataset-root", type=Path, default=Path("lerobot_conversion/outputs/robot_records_7_03_1_delta_tcp"))
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=20, help="Number of 10 Hz episode frames to replay; <=0 replays the full episode.")
    parser.add_argument("--ip", default="192.168.140.1")
    parser.add_argument("--rigid-body-id", type=int, default=1)
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--packet-channels", type=int, choices=[12, 16], default=16)
    parser.add_argument("--feedforward", choices=["pressure_model", "awac"], default="pressure_model")
    parser.add_argument("--feedback", choices=["integral_lqr", "fixed_k_integral"], default="integral_lqr")
    parser.add_argument("--pressure-checkpoint", type=Path, default=Path("motion_control_training/feedforward_pressure/runs/optimized_state12_raw_pressure/best.pt"))
    parser.add_argument("--awac-checkpoint", type=Path, default=Path("motion_control_training/KORL/runs/feedforward/awac_quadq_2k_eval_2x256/best.pt"))
    parser.add_argument(
        "--koopman-checkpoint",
        type=Path,
        default=Path("motion_control_training/koopman/runs/robot_records_7_03_1_delta_tcp_10hz_to_50hz_k50_epoch1500_wandb_online_20260706_2159/best.pt"),
    )
    parser.add_argument("--fixed-k-path", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--delta-tcp-scale", type=float, default=0.1)
    parser.add_argument("--pressure-scale", type=float, default=0.2)
    parser.add_argument("--feedback-gain-scale", type=float, default=0.05)
    parser.add_argument("--max-integral-error", type=float, default=0.5)
    parser.add_argument("--q-tcp6-weight", type=float, default=1.0)
    parser.add_argument("--q-state-tail-weight", type=float, default=0.1)
    parser.add_argument("--q-latent-weight", type=float, default=0.1)
    parser.add_argument("--q-integral-weight", type=float, default=0.5)
    parser.add_argument("--r-weight", type=float, default=10.0)
    parser.add_argument("--frequency", type=float, default=50.0)
    parser.add_argument("--log-jsonl", type=Path, default=None)
    parser.add_argument("--plot-path", type=Path, default=None)
    parser.add_argument("--output-summary", type=Path, default=None)
    args = parser.parse_args()

    if args.mock and args.hardware_enabled:
        raise SystemExit("Choose only one of --mock or --hardware-enabled.")
    if not args.mock and not args.hardware_enabled:
        raise SystemExit("Use --mock for offline wiring validation or --hardware-enabled for real hardware.")
    if args.pressure_scale < 0 or args.pressure_scale > 1.0:
        raise SystemExit("--pressure-scale must be in [0, 1] for small-amplitude real replay.")
    if args.feedback_gain_scale < 0 or args.feedback_gain_scale > 1.0:
        raise SystemExit("--feedback-gain-scale must be in [0, 1] for real replay.")

    rows = load_episode(args.dataset_root, args.episode_index)
    if not rows:
        raise SystemExit(f"episode {args.episode_index} not found")
    if args.max_frames > 0:
        rows = rows[: args.max_frames]

    first_dataset_state12 = rows[0]["state"][:12].astype(np.float32)
    if args.mock:
        state_source = MockRobotStateSource(state12=first_dataset_state12.copy(), gripper_open=float(rows[0]["state"][12]))
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
        feedforward = AwacFeedforwardAdapter(AwacFeedforwardConfig(checkpoint=args.awac_checkpoint, device=args.device))

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
    ref_gen = ReferenceGenerator(
        ReferenceGeneratorConfig(
            upper_frequency_hz=10.0,
            control_frequency_hz=args.frequency,
            delta_tcp_scale=args.delta_tcp_scale,
        )
    )
    if args.log_jsonl is not None and args.log_jsonl.exists():
        args.log_jsonl.unlink()
    logger = AsyncJsonlLogger(args.log_jsonl) if args.log_jsonl else None
    timing = TimingStats()
    safety_flags: set[str] = set()
    max_pressure = 0.0
    max_tracking_error = 0.0
    sum_abs_xyz_error = np.zeros(3, dtype=np.float64)
    sum_abs_euler_error = np.zeros(3, dtype=np.float64)
    tracking_error_samples = 0
    writes = 0
    control_steps = 0

    state_source.open()
    driver.open()
    if logger:
        logger.start()
    try:
        timer = PeriodicTimer(args.frequency)
        for upper_step, row in enumerate(rows):
            dataset_base_state12 = row["state"][:12].astype(np.float32)
            action = UpperAction(
                delta_tcp6=row["action"][:6],
                gripper_open=float(row["action"][6]),
                upper_step=upper_step,
                timestamp=row["timestamp"],
                frame_index=row["frame_index"],
                episode_index=args.episode_index,
                source="episode_replay_dataset_action",
            )
            # The reference is built from the recorded LeRobot state and action.
            # The feedback loop below always uses the live measured state.
            segment = ref_gen.build(current_state12=dataset_base_state12, action=action)
            for substep in range(segment.reference_states12.shape[0]):
                t0 = time.monotonic_ns()
                measured = state_source.read_state(blocking=True)
                reference = segment.reference_for_substep(substep)
                lifted_error = koopman.tracking_error(measured.state12, reference)
                cmd = runtime.compute(
                    current_state12=measured.state12,
                    reference_state12=reference,
                    delta_tcp6=action.delta_tcp6,
                    gripper_open=action.gripper_open,
                    lifted_error=lifted_error,
                    pressure_scale=args.pressure_scale,
                    now_ns=t0,
                    state_timestamp_ns=measured.monotonic_ns,
                    reference_timestamp_ns=t0,
                )
                writes += driver.send_physical(cmd.final_physical)
                timing.add_ns(time.monotonic_ns() - t0)
                control_steps += 1
                safety_flags.update(cmd.safety_flags)
                max_pressure = max(max_pressure, float(np.max(cmd.final_physical)))
                tcp_error = measured.state12[:6] - reference[:6]
                max_tracking_error = max(max_tracking_error, float(np.max(np.abs(tcp_error))))
                sum_abs_xyz_error += np.abs(tcp_error[:3])
                sum_abs_euler_error += np.abs(tcp_error[3:6])
                tracking_error_samples += 1
                if logger:
                    logger.log(
                        {
                            "upper_step": upper_step,
                            "substep": substep,
                            "frame_index": row["frame_index"],
                            "dataset_timestamp": row["timestamp"],
                            "dataset_base_state": dataset_base_state12.tolist(),
                            "dataset_action_delta": action.delta_tcp6.tolist(),
                            "reference_state": reference.tolist(),
                            "measured_state": measured.state12.tolist(),
                            "tracking_error_tcp6": tcp_error.tolist(),
                            "tracking_error_source": "measured_state_minus_reference_state",
                            "pressure": cmd.final_physical.tolist(),
                            "safety_flags": list(cmd.safety_flags),
                            "written_bytes_total": writes,
                        }
                    )
                if args.mock:
                    state_source.state12 = reference.copy()
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

    mean_abs_xyz_error = (
        sum_abs_xyz_error / float(tracking_error_samples) if tracking_error_samples else np.zeros(3, dtype=np.float64)
    )
    mean_abs_euler_error = (
        sum_abs_euler_error / float(tracking_error_samples) if tracking_error_samples else np.zeros(3, dtype=np.float64)
    )
    summary = {
        "mode": "mock" if args.mock else "hardware",
        "episode_index": args.episode_index,
        "upper_frames": len(rows),
        "control_steps": control_steps,
        "target_source": "lerobot_observation_state_plus_lerobot_action_delta",
        "closed_loop_state_source": "mock_perfect_tracking" if args.mock else "lumo_measured_state",
        "feedforward": args.feedforward,
        "feedback": args.feedback,
        "q_tcp6_weight": args.q_tcp6_weight,
        "q_state_tail_weight": args.q_state_tail_weight,
        "q_latent_weight": args.q_latent_weight,
        "q_integral_weight": args.q_integral_weight,
        "r_weight": args.r_weight,
        "max_pressure_physical": max_pressure,
        "max_abs_tcp6_tracking_error": max_tracking_error,
        "mean_abs_xyz_error_m": mean_abs_xyz_error.tolist(),
        "mean_abs_euler_error_rad": mean_abs_euler_error.tolist(),
        "mean_abs_xyz_error_m_avg": float(np.mean(mean_abs_xyz_error)),
        "mean_abs_euler_error_rad_avg": float(np.mean(mean_abs_euler_error)),
        "written_bytes_total": writes,
        "safety_flags": sorted(safety_flags),
        "plot_path": None if plot_path is None else str(plot_path),
        "timing": timing.summary(),
    }
    if args.output_summary:
        args.output_summary.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
