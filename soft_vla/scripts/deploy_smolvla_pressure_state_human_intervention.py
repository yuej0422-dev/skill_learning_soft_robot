from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

PROJECT_ROOT = add_src_to_path()
REPO_ROOT = PROJECT_ROOT.parent

from soft_vla.runtime.smolvla_human_intervention_runtime import (  # noqa: E402
    HumanInterventionRuntimeConfig,
    run_smolvla_human_intervention_runtime,
)


DEFAULT_CHECKPOINT = REPO_ROOT / (
    "soft_vla/outputs/full_runs/"
    "smolvla_pressure_state_bs8_20k_pressure_state_bs8_20k_20260715_161801/"
    "checkpoints/020000/pretrained_model"
)
DEFAULT_DATASET_ROOT = REPO_ROOT / "lerobot_conversion/outputs/robot_records_7_03_1_delta_tcp"
DEFAULT_KOOPMAN_CHECKPOINT = REPO_ROOT / (
    "motion_control_training/koopman/experiments/fullA_history_v2/runs/"
    "koopman_pressure16_fullA_history_v2_smoke_model_hparams_fullwindows_epoch1000_"
    "wandb_online_20260716/best.pt"
)


def validate_pressure_state_checkpoint_schema(checkpoint: Path) -> None:
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"pressure-state checkpoint config not found: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    state_shape = payload.get("input_features", {}).get("observation.state", {}).get("shape")
    action_shape = payload.get("output_features", {}).get("action", {}).get("shape")
    if state_shape != [25] or action_shape != [19]:
        raise SystemExit(
            "pressure-state intervention requires observation.state=[25] and action=[19], "
            f"got state={state_shape}, action={action_shape} from {config_path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy pressure-state SmolVLA with Xbox delta-TCP intervention and DAgger recording."
    )
    parser.add_argument("--mode", choices=["single_step", "chunk", "receding_horizon", "temporal_ensemble"], default="receding_horizon")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--real-policy", action="store_true")
    parser.add_argument("--hardware-enabled", action="store_true")
    parser.add_argument("--state-hardware-enabled", action="store_true")
    parser.add_argument("--live-observation", action="store_true")
    parser.add_argument("--camera-preview", action="store_true")
    parser.add_argument("--task", default="pick up the apple and place it on the electronic scale")
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--upper-frequency", type=float, default=10.0)
    parser.add_argument("--control-frequency", type=float, default=50.0)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument("--replan-interval", type=int, default=5)
    parser.add_argument("--chunk-trigger-margin", type=int, default=1)
    parser.add_argument("--chunk-expected-stale-steps", type=int, default=2)
    parser.add_argument("--chunk-worst-stale-steps", type=int, default=5)
    parser.add_argument("--delta-tcp-scale", type=float, default=1.0)
    parser.add_argument("--pressure-delta-scale", type=float, default=1.0)
    parser.add_argument("--pressure-scale", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", default="local/soft_robot_7_03_1_delta_tcp")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-inference-chunks", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--feedback", choices=["none", "integral_lqr", "fixed_k_integral"], default="fixed_k_integral")
    parser.add_argument("--koopman-checkpoint", type=Path, default=DEFAULT_KOOPMAN_CHECKPOINT)
    parser.add_argument(
        "--koopman-architecture",
        choices=["legacy", "fullA_history_v2"],
        default="fullA_history_v2",
    )
    parser.add_argument("--fixed-k-path", type=Path, default=None)
    parser.add_argument("--reset-integral-on-target", action="store_true")
    parser.add_argument("--feedback-gain-scale", type=float, default=0.1)
    parser.add_argument("--max-integral-error", type=float, default=0.5)
    parser.add_argument("--q-tcp6-weight", type=float, default=1.0)
    parser.add_argument("--q-state-tail-weight", type=float, default=0.1)
    parser.add_argument("--q-latent-weight", type=float, default=0.1)
    parser.add_argument("--q-integral-weight", type=float, default=0.5)
    parser.add_argument("--r-weight", type=float, default=50.0)
    parser.add_argument("--motion-policy-ready-timeout-s", type=float, default=120.0)
    parser.add_argument("--wait-for-start-key", action="store_true")
    parser.add_argument("--no-wait-for-first-action-chunk", action="store_true")
    parser.add_argument("--first-action-timeout-s", type=float, default=120.0)
    parser.add_argument("--action-print-interval-steps", type=int, default=10)
    parser.add_argument("--initial-gripper-open", type=float, default=1.0)
    parser.add_argument("--gripper-close-threshold", type=float, default=0.1)
    parser.add_argument("--gripper-open-threshold", type=float, default=0.999999)
    parser.add_argument("--episode-end-reset-sleep-s", type=float, default=7.0)
    parser.add_argument("--episode-end-reset-zero-packets", type=int, default=3)
    parser.add_argument("--ip", default="192.168.140.1")
    parser.add_argument("--rigid-body-id", type=int, default=1)
    parser.add_argument("--receive-timeout-ms", type=int, default=1000)
    parser.add_argument("--port", default="/dev/serial/by-id/usb-1a86_USB2.0-Ser_-if00-port0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--packet-channels", type=int, choices=[16], default=16)
    parser.add_argument("--zed-index", type=int, default=None)
    parser.add_argument("--zed-eye", choices=["left", "right"], default="left")
    parser.add_argument("--zed-width", type=int, default=2560)
    parser.add_argument("--zed-height", type=int, default=720)
    parser.add_argument("--zed-fps", type=int, default=30)
    parser.add_argument("--cam1-crop-right-fraction", type=float, default=0.2)
    parser.add_argument("--realsense-serial-cam2", default="401522072797")
    parser.add_argument("--realsense-serial-cam3", default="408322072769")
    parser.add_argument("--zed-warmup-usable-frames", type=int, default=10)
    parser.add_argument("--realsense-warmup-usable-frames", type=int, default=10)
    parser.add_argument("--min-realsense-mean", type=float, default=40.0)
    parser.add_argument("--camera-preview-scale", type=float, default=0.5)
    parser.add_argument("--camera-preview-fps", type=float, default=10.0)
    parser.add_argument("--camera-preview-window", default="soft_vla_pressure_state_human_live_cameras")
    parser.add_argument("--gamepad-backend", default="evdev")
    parser.add_argument("--gamepad-device-path", default=None)
    parser.add_argument("--print-gamepad-events", action="store_true")
    parser.add_argument("--gamepad-deadzone", type=float, default=0.15)
    parser.add_argument("--intervention-release-deadzone", type=float, default=0.10)
    parser.add_argument("--intervention-release-ticks", type=int, default=1)
    parser.add_argument("--human-max-delta-pos", type=float, default=0.005)
    parser.add_argument("--human-max-delta-rot", type=float, default=0.025)
    parser.add_argument("--rotation-enabled", action="store_true")
    parser.add_argument("--rotation-axis", choices=["none", "roll", "pitch", "yaw", "pitch_yaw"], default="pitch_yaw")
    parser.add_argument("--no-human-target-integration", action="store_true")
    parser.add_argument("--human-target-max-pos-offset", type=float, default=0.02)
    parser.add_argument("--human-target-max-rot-offset", type=float, default=0.05)
    parser.add_argument("--handover-blend-steps", type=int, default=2)
    parser.add_argument("--blend-tcp-only", action="store_true")
    parser.add_argument("--blend-gripper", action="store_true")
    parser.add_argument("--remote-control-debug", action="store_true")
    parser.add_argument("--episode-save-root", type=Path, default=REPO_ROOT / "soft_vla/tests/tmp/pressure_state_human_intervention_episodes")
    parser.add_argument("--no-save-human-episodes", action="store_true")
    parser.add_argument("--log-jsonl", type=Path, default=None)
    args = parser.parse_args()

    if args.mock and args.hardware_enabled:
        raise SystemExit("Choose only one of --mock or --hardware-enabled.")
    if args.mock and args.real_policy:
        raise SystemExit("Choose only one of --mock or --real-policy.")
    if not args.mock and not args.real_policy and not args.hardware_enabled:
        raise SystemExit("Use --mock, --real-policy, or --hardware-enabled.")
    if args.control_frequency != 50.0:
        raise SystemExit("pressure-state intervention requires --control-frequency 50")
    if args.upper_frequency <= 0 or args.control_frequency <= 0:
        raise SystemExit("frequencies must be positive")
    if args.pressure_delta_scale <= 0:
        raise SystemExit("--pressure-delta-scale must be positive for aligned action19 recording")
    if not (0.0 <= args.pressure_scale <= 1.0):
        raise SystemExit("--pressure-scale must be in [0, 1]")
    if not (0.0 <= args.feedback_gain_scale <= 1.0):
        raise SystemExit("--feedback-gain-scale must be in [0, 1]")
    if not (0.0 <= args.cam1_crop_right_fraction < 1.0):
        raise SystemExit("--cam1-crop-right-fraction must be in [0, 1)")
    if not (0.0 <= args.gripper_close_threshold < args.gripper_open_threshold <= 1.0):
        raise SystemExit("gripper thresholds must satisfy 0 <= close < open <= 1")

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else REPO_ROOT / path

    checkpoint = resolve(args.checkpoint)
    validate_pressure_state_checkpoint_schema(checkpoint)
    report = run_smolvla_human_intervention_runtime(
        HumanInterventionRuntimeConfig(
            duration_s=args.duration_s,
            upper_frequency_hz=args.upper_frequency,
            control_frequency_hz=args.control_frequency,
            mode=args.mode,
            vla_action_mode="pressure_delta19",
            reference_interpolation="zero_order_hold",
            chunk_size=args.chunk_size,
            execution_horizon=args.execution_horizon,
            replan_interval=args.replan_interval,
            chunk_trigger_margin=args.chunk_trigger_margin,
            chunk_expected_stale_steps=args.chunk_expected_stale_steps,
            chunk_worst_stale_steps=args.chunk_worst_stale_steps,
            delta_tcp_scale=args.delta_tcp_scale,
            pressure_delta_scale=args.pressure_delta_scale,
            pressure_scale=args.pressure_scale,
            log_jsonl=None if args.log_jsonl is None else str(resolve(args.log_jsonl)),
            hardware_enabled=args.hardware_enabled,
            state_hardware_enabled=args.state_hardware_enabled,
            ip=args.ip,
            rigid_body_id=args.rigid_body_id,
            receive_timeout_ms=args.receive_timeout_ms,
            port=args.port,
            baudrate=args.baudrate,
            packet_channels=args.packet_channels,
            mock=args.mock or not args.hardware_enabled,
            real_policy=args.real_policy,
            live_observation=args.live_observation,
            camera_preview=args.camera_preview,
            task=args.task,
            zed_index=args.zed_index,
            zed_eye=args.zed_eye,
            zed_width=args.zed_width,
            zed_height=args.zed_height,
            zed_fps=args.zed_fps,
            cam1_crop_right_fraction=args.cam1_crop_right_fraction,
            realsense_serial_cam2=args.realsense_serial_cam2,
            realsense_serial_cam3=args.realsense_serial_cam3,
            zed_warmup_usable_frames=args.zed_warmup_usable_frames,
            realsense_warmup_usable_frames=args.realsense_warmup_usable_frames,
            min_realsense_mean=args.min_realsense_mean,
            camera_preview_scale=args.camera_preview_scale,
            camera_preview_fps=args.camera_preview_fps,
            camera_preview_window=args.camera_preview_window,
            checkpoint=str(checkpoint),
            dataset_root=str(resolve(args.dataset_root)),
            repo_id=args.repo_id,
            episode_index=args.episode_index,
            video_backend=args.video_backend,
            device=args.device,
            use_amp=not args.no_amp,
            max_inference_chunks=args.max_inference_chunks,
            feedforward="external",
            feedback=args.feedback,
            koopman_checkpoint=str(resolve(args.koopman_checkpoint)),
            koopman_architecture=args.koopman_architecture,
            fixed_k_path=None if args.fixed_k_path is None else str(resolve(args.fixed_k_path)),
            reset_integral_on_target=args.reset_integral_on_target,
            feedback_gain_scale=args.feedback_gain_scale,
            max_integral_error=args.max_integral_error,
            q_tcp6_weight=args.q_tcp6_weight,
            q_state_tail_weight=args.q_state_tail_weight,
            q_latent_weight=args.q_latent_weight,
            q_integral_weight=args.q_integral_weight,
            r_weight=args.r_weight,
            motion_policy_ready_timeout_s=args.motion_policy_ready_timeout_s,
            wait_for_start_key=args.wait_for_start_key,
            wait_for_first_action_chunk=not args.no_wait_for_first_action_chunk,
            first_action_timeout_s=args.first_action_timeout_s,
            action_print_interval_steps=args.action_print_interval_steps,
            initial_gripper_open=args.initial_gripper_open,
            gripper_close_threshold=args.gripper_close_threshold,
            gripper_open_threshold=args.gripper_open_threshold,
            episode_end_reset_sleep_s=args.episode_end_reset_sleep_s,
            episode_end_reset_zero_packets=args.episode_end_reset_zero_packets,
            gamepad_backend=args.gamepad_backend,
            gamepad_device_path=args.gamepad_device_path,
            print_gamepad_events=args.print_gamepad_events,
            joystick_deadzone=args.gamepad_deadzone,
            intervention_release_deadzone=args.intervention_release_deadzone,
            intervention_release_ticks=args.intervention_release_ticks,
            human_max_delta_pos=args.human_max_delta_pos,
            human_max_delta_rot=args.human_max_delta_rot,
            rotation_enabled=args.rotation_enabled,
            rotation_axis=args.rotation_axis,
            human_target_integration=not args.no_human_target_integration,
            human_target_max_pos_offset=args.human_target_max_pos_offset,
            human_target_max_rot_offset=args.human_target_max_rot_offset,
            handover_blend_steps=args.handover_blend_steps,
            blend_tcp_only=args.blend_tcp_only,
            blend_gripper=args.blend_gripper,
            remote_control_debug=args.remote_control_debug,
            episode_save_root=str(resolve(args.episode_save_root)),
            save_human_episodes=not args.no_save_human_episodes,
            save_pressure_state_training_data=True,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
