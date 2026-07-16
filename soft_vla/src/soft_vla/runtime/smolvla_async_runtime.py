from __future__ import annotations

import json
import multiprocessing as mp
import queue
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from soft_vla.inference.chunk_execution.registry import make_chunk_executor
from soft_vla.motion_control.policy_factory import MotionPolicyConfig, build_motion_policy
from soft_vla.motion_control.reference_generator import ReferenceGenerator, ReferenceGeneratorConfig
from soft_vla.real_robot.live_cameras import LiveCameraConfig, LiveThreeCameraSource
from soft_vla.real_robot.pressure_driver import MockPressureDriver, SerialPressureDriver, SerialPressureDriverConfig
from soft_vla.real_robot.robot_io import LuMoStateSource, LuMoStateSourceConfig, MockRobotStateSource
from soft_vla.runtime.shared_state import UpperAction
from soft_vla.runtime.timing import PeriodicTimer, TimingStats


STOP = "__STOP__"


@dataclass(frozen=True)
class SmolVLAAsyncRuntimeConfig:
    duration_s: float = 2.0
    upper_frequency_hz: float = 10.0
    control_frequency_hz: float = 50.0
    chunk_size: int = 50
    execution_horizon: int = 10
    replan_interval: int = 5
    chunk_trigger_margin: int = 1
    chunk_expected_stale_steps: int = 2
    chunk_worst_stale_steps: int = 5
    mode: str = "receding_horizon"
    delta_tcp_scale: float = 0.2
    pressure_scale: float = 0.2
    log_jsonl: str | None = None
    hardware_enabled: bool = False
    state_hardware_enabled: bool = False
    ip: str = "192.168.140.1"
    rigid_body_id: int = 1
    receive_timeout_ms: int = 1000
    port: str = "/dev/serial/by-id/usb-1a86_USB2.0-Ser_-if00-port0"
    baudrate: int = 115200
    packet_channels: int = 16
    mock: bool = True
    real_policy: bool = False
    vla_backend: str = "smolvla"
    live_observation: bool = False
    task: str = "pick up the apple and place it on the electronic scale"
    zed_index: int | None = None
    zed_eye: str = "left"
    zed_width: int = 2560
    zed_height: int = 720
    zed_fps: int = 30
    realsense_serial_cam2: str | None = None
    realsense_serial_cam3: str | None = None
    zed_warmup_usable_frames: int = 10
    realsense_warmup_usable_frames: int = 10
    min_realsense_mean: float = 40.0
    camera_preview: bool = False
    camera_preview_scale: float = 0.5
    camera_preview_fps: float = 10.0
    camera_preview_window: str = "soft_vla_live_cameras"
    checkpoint: str | None = None
    dataset_root: str | None = None
    repo_id: str = "local/soft_robot_7_03_1_delta_tcp"
    episode_index: int | None = 0
    video_backend: str = "pyav"
    device: str = "cuda"
    use_amp: bool = True
    max_inference_chunks: int | None = None
    feedforward: str = "pressure_model"
    feedback: str = "fixed_k_integral"
    pressure_checkpoint: str | None = None
    awac_checkpoint: str | None = None
    koopman_checkpoint: str | None = None
    fixed_k_path: str | None = None
    feedback_gain_scale: float = 0.1
    max_integral_error: float = 0.5
    q_tcp6_weight: float = 1.0
    q_state_tail_weight: float = 0.1
    q_latent_weight: float = 0.1
    q_integral_weight: float = 0.5
    r_weight: float = 50.0
    safe_zero_packets: int = 3
    safe_zero_interval_s: float = 0.03
    episode_end_reset_sleep_s: float = 7.0
    episode_end_reset_zero_packets: int = 3
    motion_policy_ready_timeout_s: float = 120.0
    wait_for_start_key: bool = False
    action_print_interval_steps: int = 10
    initial_gripper_open: float = 1.0
    gripper_close_threshold: float = 0.2
    gripper_open_threshold: float = 0.8
    wait_for_first_action_chunk: bool = True
    first_action_timeout_s: float = 120.0


def run_mock_smolvla_async_runtime(config: SmolVLAAsyncRuntimeConfig) -> dict[str, Any]:
    """Run the four-process VLA deployment skeleton.

    The 50 Hz process always goes through the motion-policy stack. Hardware is
    selected only by ``config.hardware_enabled``; the inference backend is kept
    behind a process boundary so another VLA can replace SmolVLA later.
    """

    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    motion_policy_ready = ctx.Event()
    vla_policy_ready = ctx.Event()
    run_start = ctx.Event()
    first_action_chunk_ready = ctx.Event()
    reference_queue: mp.Queue = ctx.Queue(maxsize=16)
    upper_state_queue: mp.Queue = ctx.Queue(maxsize=32)
    inference_state_queue: mp.Queue = ctx.Queue(maxsize=32)
    inference_request_queue: mp.Queue = ctx.Queue(maxsize=1)
    chunk_queue: mp.Queue = ctx.Queue(maxsize=8)
    preview_queue: mp.Queue | None = ctx.Queue(maxsize=2) if config.camera_preview and config.live_observation else None
    log_queue: mp.Queue = ctx.Queue(maxsize=4096)
    summary_queue: mp.Queue = ctx.Queue()

    if config.log_jsonl:
        path = Path(config.log_jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()

    processes = [
        ctx.Process(
            name="soft-vla-control-50hz",
            target=control_process_50hz,
            args=(
                config,
                stop_event,
                motion_policy_ready,
                run_start,
                first_action_chunk_ready,
                reference_queue,
                upper_state_queue,
                inference_state_queue,
                log_queue,
                summary_queue,
            ),
        ),
        ctx.Process(
            name="soft-vla-upper-10hz",
            target=upper_dispatch_process_10hz,
            args=(
                config,
                stop_event,
                run_start,
                first_action_chunk_ready,
                reference_queue,
                upper_state_queue,
                inference_request_queue,
                chunk_queue,
                log_queue,
                summary_queue,
            ),
        ),
        ctx.Process(
            name="soft-vla-smolvla-inference",
            target=smolvla_inference_process,
            args=(
                config,
                stop_event,
                motion_policy_ready,
                vla_policy_ready,
                run_start,
                first_action_chunk_ready,
                inference_state_queue,
                inference_request_queue,
                chunk_queue,
                log_queue,
                summary_queue,
                preview_queue,
            ),
        ),
        ctx.Process(
            name="soft-vla-async-logger",
            target=logger_process,
            args=(config, stop_event, log_queue, summary_queue),
        ),
    ]
    if preview_queue is not None:
        processes.append(
            ctx.Process(
                name="soft-vla-camera-preview",
                target=camera_preview_process,
                args=(config, stop_event, preview_queue, summary_queue),
            )
        )
    previous_signal_handlers, signal_state = _install_stop_signal_handlers(stop_event)
    for process in processes:
        process.start()

    try:
        if config.wait_for_start_key:
            _wait_for_ready_events(
                config,
                stop_event,
                [("motion_policy", motion_policy_ready), ("vla_policy", vla_policy_ready)],
            )
            if not stop_event.is_set():
                print("[soft_vla] motion policy and SmolVLA weights are ready. Press any key to start execution.", flush=True)
                _wait_for_keypress()
                print("[soft_vla] start signal received; entering runtime loops.", flush=True)
                run_start.set()
        else:
            run_start.set()
        _wait_for_first_action_chunk_if_needed(config, stop_event, first_action_chunk_ready, "main")
        deadline = time.monotonic() + float(config.duration_s)
        while time.monotonic() < deadline and not stop_event.is_set():
            time.sleep(0.05)
    finally:
        stop_event.set()
        _best_effort_put(log_queue, STOP)
        if preview_queue is not None:
            _best_effort_put(preview_queue, STOP)
        for process in processes:
            process.join(timeout=5.0)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
        _restore_signal_handlers(previous_signal_handlers)

    summaries: dict[str, Any] = {}
    while True:
        try:
            item = summary_queue.get_nowait()
        except queue.Empty:
            break
        summaries[item["name"]] = item["summary"]
    exitcodes = {process.name: process.exitcode for process in processes}
    interrupted_by = signal_state.get("name")
    ok = all(code == 0 for code in exitcodes.values()) and interrupted_by is None
    return {"ok": ok, "interrupted_by": interrupted_by, "exitcodes": exitcodes, "summaries": summaries}


def run_smolvla_async_runtime(config: SmolVLAAsyncRuntimeConfig) -> dict[str, Any]:
    return run_mock_smolvla_async_runtime(config)


def control_process_50hz(
    config: SmolVLAAsyncRuntimeConfig,
    stop_event,
    motion_policy_ready,
    run_start,
    first_action_chunk_ready,
    reference_queue,
    upper_state_queue,
    inference_state_queue,
    log_queue,
    summary_queue,
) -> None:
    _install_stop_signal_handlers(stop_event)
    timing = TimingStats()
    motion_policy = build_motion_policy(
        MotionPolicyConfig(
            feedforward=config.feedforward,
            feedback=config.feedback,
            pressure_checkpoint=config.pressure_checkpoint or MotionPolicyConfig.pressure_checkpoint,
            awac_checkpoint=config.awac_checkpoint or MotionPolicyConfig.awac_checkpoint,
            koopman_checkpoint=config.koopman_checkpoint or MotionPolicyConfig.koopman_checkpoint,
            fixed_k_path=config.fixed_k_path,
            device=config.device,
            feedback_gain_scale=config.feedback_gain_scale,
            max_integral_error=config.max_integral_error,
            q_tcp6_weight=config.q_tcp6_weight,
            q_state_tail_weight=config.q_state_tail_weight,
            q_latent_weight=config.q_latent_weight,
            q_integral_weight=config.q_integral_weight,
            r_weight=config.r_weight,
        )
    )
    print(f"[soft_vla] motion policy initialized: {_format_motion_policy_metadata(motion_policy.metadata)}", flush=True)
    _best_effort_put(
        log_queue,
        {
            "process": "control_50hz",
            "event": "motion_policy_initialized",
            "motion_policy": motion_policy.metadata,
        },
    )
    motion_policy_ready.set()
    state_hardware_enabled = bool(config.hardware_enabled or config.state_hardware_enabled)
    estimated_gripper_open = 1.0 if float(config.initial_gripper_open) >= 0.5 else 0.0
    if state_hardware_enabled:
        state_source = LuMoStateSource(
            LuMoStateSourceConfig(
                ip=config.ip,
                rigid_body_id=config.rigid_body_id,
                receive_timeout_ms=config.receive_timeout_ms,
            )
        )
    else:
        state_source = MockRobotStateSource(state12=_mock_state12(0, config.control_frequency_hz), gripper_open=1.0)
    if config.hardware_enabled:
        driver = SerialPressureDriver(
            SerialPressureDriverConfig(
                port=config.port,
                baudrate=config.baudrate,
                packet_channels=config.packet_channels,
            )
        )
    else:
        driver = MockPressureDriver(packet_channels=config.packet_channels)
    latest_reference: dict[str, Any] | None = None
    steps = 0
    underruns = 0
    writes = 0
    safety_flags: set[str] = set()
    max_pressure = 0.0
    sum_abs_xyz_error = np.zeros(3, dtype=np.float64)
    sum_abs_euler_error = np.zeros(3, dtype=np.float64)
    tracking_error_samples = 0
    safe_exit_zero_packets = 0
    safe_exit_zero_errors: list[str] = []
    episode_end_reset_packets = 0
    episode_end_reset_errors: list[str] = []
    episode_end_reset_requested = False
    try:
        _wait_for_run_start(stop_event, run_start)
        if stop_event.is_set():
            return
        state_source.open()
        if config.wait_for_first_action_chunk:
            first_action_deadline = time.monotonic() + float(config.first_action_timeout_s)
            prestart_timer = PeriodicTimer(config.control_frequency_hz)
            prestart_steps = 0
            while not stop_event.is_set() and not first_action_chunk_ready.is_set():
                if time.monotonic() >= first_action_deadline:
                    raise TimeoutError(
                        f"control_50hz timed out waiting for first action chunk after {config.first_action_timeout_s:.1f}s"
                    )
                measured = state_source.read_state(blocking=True)
                state_msg = {
                    "step": prestart_steps,
                    "monotonic_ns": measured.monotonic_ns,
                    "state12": measured.state12.tolist(),
                    "gripper_open": estimated_gripper_open,
                    "measured_gripper_open": measured.gripper_open,
                    "source": measured.source,
                    "prestart": True,
                }
                _best_effort_put(upper_state_queue, state_msg)
                _best_effort_put(inference_state_queue, state_msg)
                prestart_steps += 1
                prestart_timer.wait_next()
        if stop_event.is_set():
            return
        if config.wait_for_first_action_chunk:
            print("[soft_vla] first action chunk ready; starting 50Hz control loop.", flush=True)
        driver.open()
        timer = PeriodicTimer(config.control_frequency_hz)
        while not stop_event.is_set():
            t0 = time.monotonic_ns()
            reset_request: dict[str, Any] | None = None
            while True:
                try:
                    reference_msg = reference_queue.get_nowait()
                except queue.Empty:
                    break
                if isinstance(reference_msg, dict) and reference_msg.get("event") == "episode_end_reset_zero":
                    reset_request = reference_msg
                else:
                    latest_reference = reference_msg
            if reset_request is not None:
                episode_end_reset_requested = True
                hold_s = max(0.0, float(reset_request.get("hold_s", config.episode_end_reset_sleep_s)))
                attempts = max(1, int(reset_request.get("zero_packets", config.episode_end_reset_zero_packets)))
                episode_end_reset_packets, episode_end_reset_errors = _send_zero_pressure_safely(
                    driver,
                    attempts=attempts,
                    interval_s=config.safe_zero_interval_s,
                )
                print(
                    f"[SAFETY] episode end reset: sent {episode_end_reset_packets} zero-pressure packets; holding {hold_s:.1f}s",
                    flush=True,
                )
                _best_effort_put(
                    log_queue,
                    {
                        "process": "control_50hz",
                        "event": "episode_end_reset_zero_pressure",
                        "zero_packets": episode_end_reset_packets,
                        "errors": episode_end_reset_errors,
                        "hold_s": hold_s,
                        "hardware_enabled": config.hardware_enabled,
                    },
                )
                if hold_s > 0.0:
                    time.sleep(hold_s)
                latest_reference = None
                timer.next_deadline_ns = time.monotonic_ns() + timer.period_ns
                continue
            measured = state_source.read_state(blocking=True)
            state_msg = {
                "step": steps,
                "monotonic_ns": measured.monotonic_ns,
                "state12": measured.state12.tolist(),
                "gripper_open": estimated_gripper_open,
                "measured_gripper_open": measured.gripper_open,
                "source": measured.source,
            }
            if latest_reference is None:
                underruns += 1
                command = motion_policy.runtime.safety.build_pressure_command(
                    motion_norm12=np.zeros(12, dtype=np.float32),
                    gripper_open=estimated_gripper_open,
                    now_ns=t0,
                    state_timestamp_ns=measured.monotonic_ns,
                    current_state12=measured.state12,
                    pressure_scale=config.pressure_scale,
                )
                ref_step = None
                reference_state = None
                tcp_error = None
            else:
                refs = np.asarray(latest_reference["reference_states12"], dtype=np.float32)
                substep = min(max(steps - int(latest_reference["control_start_step"]), 0), refs.shape[0] - 1)
                reference = refs[substep]
                delta_tcp6 = np.asarray(latest_reference["delta_tcp6"], dtype=np.float32)
                estimated_gripper_open = 1.0 if float(latest_reference["gripper_open"]) >= 0.5 else 0.0
                lifted_error = motion_policy.koopman.tracking_error(measured.state12, reference)
                command = motion_policy.runtime.compute(
                    current_state12=measured.state12,
                    reference_state12=reference,
                    delta_tcp6=delta_tcp6,
                    gripper_open=estimated_gripper_open,
                    lifted_error=lifted_error,
                    pressure_scale=config.pressure_scale,
                    now_ns=t0,
                    state_timestamp_ns=measured.monotonic_ns,
                    reference_timestamp_ns=t0,
                )
                ref_step = {"upper_step": latest_reference["upper_step"], "substep": int(substep)}
                reference_state = reference
                tcp_error = measured.state12[:6] - reference[:6]
                sum_abs_xyz_error += np.abs(tcp_error[:3])
                sum_abs_euler_error += np.abs(tcp_error[3:6])
                tracking_error_samples += 1
                if not config.hardware_enabled and isinstance(state_source, MockRobotStateSource):
                    state_source.state12 = reference.copy()
                    state_source.gripper_open = estimated_gripper_open
            writes += driver.send_physical(command.final_physical)
            timing.add_ns(time.monotonic_ns() - t0)
            safety_flags.update(command.safety_flags)
            max_pressure = max(max_pressure, float(np.max(command.final_physical)))
            state_msg["motion_norm12"] = command.motion_norm12.tolist()
            state_msg["pressure16"] = command.final_physical.tolist()
            state_msg["u_p12"] = command.motion_norm12.tolist()
            state_msg["u_paw4"] = command.final_physical[12:16].tolist()
            _best_effort_put(upper_state_queue, state_msg)
            _best_effort_put(inference_state_queue, state_msg)
            _best_effort_put(
                log_queue,
                {
                    "process": "control_50hz",
                    "step": steps,
                    "reference": ref_step,
                    "measured_state": measured.state12.tolist(),
                    "reference_state": None if reference_state is None else reference_state.tolist(),
                    "tracking_error_tcp6": None if tcp_error is None else tcp_error.tolist(),
                    "motion_norm12": command.motion_norm12.tolist(),
                    "pressure": command.final_physical.tolist(),
                    "flags": list(command.safety_flags),
                    "written_bytes_total": writes,
                    "hardware_enabled": config.hardware_enabled,
                    "state_hardware_enabled": state_hardware_enabled,
                    "gripper_open_estimated": estimated_gripper_open,
                    "motion_policy": motion_policy.metadata,
                },
            )
            steps += 1
            timer.wait_next()
    finally:
        try:
            safe_exit_zero_packets, safe_exit_zero_errors = _send_zero_pressure_safely(
                driver,
                attempts=config.safe_zero_packets,
                interval_s=config.safe_zero_interval_s,
            )
            _best_effort_put(
                log_queue,
                {
                    "process": "control_50hz",
                    "event": "safe_exit_zero_pressure",
                    "zero_packets": safe_exit_zero_packets,
                    "errors": safe_exit_zero_errors,
                    "hardware_enabled": config.hardware_enabled,
                },
            )
        finally:
            driver.close()
            state_source.close()
    mean_abs_xyz_error = (
        sum_abs_xyz_error / float(tracking_error_samples) if tracking_error_samples else np.zeros(3, dtype=np.float64)
    )
    mean_abs_euler_error = (
        sum_abs_euler_error / float(tracking_error_samples) if tracking_error_samples else np.zeros(3, dtype=np.float64)
    )
    summary_queue.put(
        {
            "name": "control_50hz",
            "summary": {
                "steps": steps,
                "underruns": underruns,
                "written_bytes_total": writes,
                "max_pressure_physical": max_pressure,
                "mean_abs_xyz_error_m": mean_abs_xyz_error.tolist(),
                "mean_abs_euler_error_rad": mean_abs_euler_error.tolist(),
                "safety_flags": sorted(safety_flags),
                "hardware_enabled": config.hardware_enabled,
                "state_hardware_enabled": state_hardware_enabled,
                "safe_exit_zero_packets": safe_exit_zero_packets,
                "safe_exit_zero_errors": safe_exit_zero_errors,
                "episode_end_reset_requested": episode_end_reset_requested,
                "episode_end_reset_packets": episode_end_reset_packets,
                "episode_end_reset_errors": episode_end_reset_errors,
                "motion_policy": motion_policy.metadata,
                "timing": timing.summary(),
            },
        }
    )


def upper_dispatch_process_10hz(
    config: SmolVLAAsyncRuntimeConfig,
    stop_event,
    run_start,
    first_action_chunk_ready,
    reference_queue,
    state_queue,
    request_queue,
    chunk_queue,
    log_queue,
    summary_queue,
) -> None:
    _install_stop_signal_handlers(stop_event)
    _wait_for_run_start(stop_event, run_start)
    _wait_for_first_action_chunk_if_needed(config, stop_event, first_action_chunk_ready, "upper_10hz")
    if stop_event.is_set():
        return
    timer = PeriodicTimer(config.upper_frequency_hz)
    ref_gen = ReferenceGenerator(
        ReferenceGeneratorConfig(
            upper_frequency_hz=config.upper_frequency_hz,
            control_frequency_hz=config.control_frequency_hz,
            delta_tcp_scale=config.delta_tcp_scale,
        )
    )
    executor = make_chunk_executor(
        {
            "mode": config.mode,
            "chunk_size": config.chunk_size,
            "execution_horizon": config.execution_horizon,
            "replan_interval": config.replan_interval,
            "chunk_trigger_margin": config.chunk_trigger_margin,
            "chunk_expected_stale_steps": config.chunk_expected_stale_steps,
        }
    )
    latest_state = {"state12": np.zeros(12, dtype=np.float32).tolist(), "gripper_open": 1.0}
    steps = 0
    fallbacks = 0
    inference_running = True
    replan_triggers = 0
    period_timing = TimingStats()
    last_step_start_ns: int | None = None
    while not stop_event.is_set():
        t0 = time.monotonic_ns()
        period_ms = None if last_step_start_ns is None else (t0 - last_step_start_ns) / 1_000_000.0
        last_step_start_ns = t0
        if period_ms is not None:
            period_timing.samples_ms.append(period_ms)
        while True:
            try:
                latest_state = state_queue.get_nowait()
            except queue.Empty:
                break
        while True:
            try:
                chunk_msg = chunk_queue.get_nowait()
            except queue.Empty:
                break
            executor.submit_chunk(
                chunk_msg["chunk"],
                observation_timestamp=float(chunk_msg["start_step"]),
                inference_start_timestamp=float(chunk_msg["inference_start_ns"]),
                inference_end_timestamp=float(chunk_msg["inference_end_ns"]),
                request_tick=int(chunk_msg.get("request_tick", chunk_msg["start_step"])),
                result_tick=int(chunk_msg.get("result_tick", chunk_msg["start_step"])),
                next_dispatch_tick=steps,
                drop_stale_actions=True,
            )
            inference_running = False
        record_source = "exception_fallback"
        try:
            record = executor.get_action(steps, time.monotonic())
            action7 = record.action
            record_source = record.source
            if "fallback" in record.source:
                fallbacks += 1
                action7 = action7.copy()
                action7[6] = float(latest_state.get("gripper_open", 1.0))
        except Exception:
            fallbacks += 1
            action7 = np.zeros(7, dtype=np.float32)
            action7[6] = float(latest_state.get("gripper_open", 1.0))
        upper_action = UpperAction(
            delta_tcp6=action7[:6],
            gripper_open=float(action7[6]),
            upper_step=steps,
            source="smolvla_async",
        )
        segment = ref_gen.build(current_state12=np.asarray(latest_state["state12"], dtype=np.float32), action=upper_action)
        _drain_put(
            reference_queue,
            {
                "upper_step": steps,
                "control_start_step": segment.control_start_step,
                "reference_states12": segment.reference_states12.tolist(),
                "gripper_open": segment.gripper_open,
                "delta_tcp6": upper_action.delta_tcp6.tolist(),
            },
        )
        if config.action_print_interval_steps > 0 and steps % int(config.action_print_interval_steps) == 0:
            period_text = "nan" if period_ms is None else f"{period_ms:.3f}"
            print(
                "[soft_vla] upper_step={step} period_ms={period} source={source} "
                "action=[dx={a0:.6g}, dy={a1:.6g}, dz={a2:.6g}, "
                "droll={a3:.6g}, dpitch={a4:.6g}, dyaw={a5:.6g}, gripper={a6:.3g}]".format(
                    step=steps,
                    period=period_text,
                    source=record_source,
                    a0=float(action7[0]),
                    a1=float(action7[1]),
                    a2=float(action7[2]),
                    a3=float(action7[3]),
                    a4=float(action7[4]),
                    a5=float(action7[5]),
                    a6=float(action7[6]),
                ),
                flush=True,
            )
        replan_triggered = False
        if executor.needs_replan(steps, time.monotonic()) and not inference_running:
            request = {
                "request_tick": steps + 1,
                "request_time_ns": time.monotonic_ns(),
                "mode": config.mode,
                "reason": "executor_needs_replan",
            }
            try:
                request_queue.put_nowait(request)
                inference_running = True
                replan_triggered = True
                replan_triggers += 1
            except queue.Full:
                inference_running = True
        _best_effort_put(
            log_queue,
            {
                "process": "upper_10hz",
                "mode": config.mode,
                "step": steps,
                "tick": steps,
                "wall_time": t0 / 1_000_000_000.0,
                "action_dispatch_time": time.monotonic_ns() / 1_000_000_000.0,
                "action": action7.astype(float).tolist(),
                "action_before_clip": action7.astype(float).tolist(),
                "action_after_clip": action7.astype(float).tolist(),
                "record_source": record_source,
                "period_ms": period_ms,
                "request_tick": record.debug.get("request_tick"),
                "result_tick": record.debug.get("result_tick"),
                "effective_tick": record.debug.get("effective_tick"),
                "next_dispatch_tick": steps,
                "stale_steps": record.debug.get("stale_steps"),
                "chunk_id": record.chunk_id,
                "chunk_start_tick": record.debug.get("request_tick"),
                "valid_start_tick": record.debug.get("valid_start_tick", record.debug.get("effective_tick")),
                "chunk_local_idx": record.chunk_step,
                "selected_action_idx": record.chunk_step,
                "queue_length": executor.get_debug_state().get("queue_len"),
                "queue_underflow": "fallback" in record_source,
                "fallback_used": "fallback" in record_source or record_source == "exception_fallback",
                "replan_triggered": replan_triggered,
                "inference_running": inference_running,
                "pending_future_count": executor.get_debug_state().get("queue_len"),
                "te_num_candidates": record.debug.get("te_num_candidates"),
                "te_candidate_chunk_ids": record.debug.get("te_candidate_chunk_ids"),
                "te_candidate_local_indices": record.debug.get("te_candidate_local_indices"),
                "te_weights": record.debug.get("weights"),
                "executor": executor.get_debug_state(),
            },
        )
        steps += 1
        timer.wait_next()
    summary_queue.put(
        {
            "name": "upper_10hz",
            "summary": {
                "steps": steps,
                "fallbacks": fallbacks,
                "replan_triggers": replan_triggers,
                "period_timing": period_timing.summary(),
            },
        }
    )


def smolvla_inference_process(
    config: SmolVLAAsyncRuntimeConfig,
    stop_event,
    motion_policy_ready,
    vla_policy_ready,
    run_start,
    first_action_chunk_ready,
    state_queue,
    request_queue,
    chunk_queue,
    log_queue,
    summary_queue,
    preview_queue=None,
    episode_observation_queue=None,
) -> None:
    _install_stop_signal_handlers(stop_event)
    _wait_for_motion_policy_ready(config, stop_event, motion_policy_ready)
    if config.real_policy:
        smolvla_real_inference_process(
            config,
            stop_event,
            vla_policy_ready,
            run_start,
            first_action_chunk_ready,
            state_queue,
            request_queue,
            chunk_queue,
            log_queue,
            summary_queue,
            preview_queue,
            episode_observation_queue,
        )
        return
    vla_policy_ready.set()
    _wait_for_run_start(stop_event, run_start)
    if stop_event.is_set():
        return
    chunks = 0
    timing = TimingStats()
    pending_request: dict[str, Any] | None = _make_initial_inference_request()
    while not stop_event.is_set():
        if pending_request is None:
            pending_request = _get_next_inference_request(request_queue, stop_event)
            if pending_request is None:
                continue
        request_tick = int(pending_request["request_tick"])
        request_time_ns = int(pending_request["request_time_ns"])
        t0 = time.monotonic_ns()
        chunk = np.zeros((config.chunk_size, 7), dtype=np.float32)
        chunk[:, 6] = 1.0
        # Tiny deterministic motion signal lets queue/reference plumbing be inspected.
        chunk[:, 0] = 0.0002 * np.sin(np.linspace(0.0, np.pi, config.chunk_size, dtype=np.float32))
        end_ns = time.monotonic_ns()
        result_tick = _result_tick_for_request(config, pending_request, end_ns)
        _drain_put(
            chunk_queue,
            {
                "chunk_id": chunks,
                "start_step": request_tick,
                "request_tick": request_tick,
                "request_time_ns": request_time_ns,
                "result_tick": result_tick,
                "result_time_ns": end_ns,
                "chunk": chunk,
                "inference_start_ns": t0,
                "inference_end_ns": end_ns,
            },
        )
        if chunks == 0:
            first_action_chunk_ready.set()
            print("[soft_vla] first action chunk ready: backend=mock", flush=True)
        timing.add_ns(end_ns - t0)
        _best_effort_put(
            log_queue,
            {
                "process": "smolvla_inference",
                "chunk_id": chunks,
                "chunk_size": config.chunk_size,
                "request_tick": request_tick,
                "request_time": request_time_ns / 1_000_000_000.0,
                "result_tick": result_tick,
                "result_time": end_ns / 1_000_000_000.0,
                "inference_latency_ms": (end_ns - request_time_ns) / 1_000_000.0,
                "model_latency_ms": (end_ns - t0) / 1_000_000.0,
                "bootstrap": bool(pending_request.get("bootstrap", False)),
            },
        )
        chunks += 1
        pending_request = None
    summary_queue.put({"name": "smolvla_inference", "summary": {"chunks": chunks, "timing": timing.summary()}})


def smolvla_real_inference_process(
    config: SmolVLAAsyncRuntimeConfig,
    stop_event,
    vla_policy_ready,
    run_start,
    first_action_chunk_ready,
    state_queue,
    request_queue,
    chunk_queue,
    log_queue,
    summary_queue,
    preview_queue=None,
    episode_observation_queue=None,
) -> None:
    import os
    import threading

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    from soft_vla.data.replay_source import LeRobotReplaySource

    if config.checkpoint is None:
        raise ValueError("checkpoint is required when real_policy=True")
    if config.dataset_root is None:
        raise ValueError("dataset_root is required when real_policy=True")

    checkpoint = Path(config.checkpoint)
    device = config.device
    policy = SmolVLAPolicy.from_pretrained(checkpoint, local_files_only=True)
    policy.config.device = device
    policy.to(device)
    policy.eval()
    policy.reset()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    print(f"[soft_vla] SmolVLA weights initialized: checkpoint={checkpoint}, device={device}", flush=True)
    vla_policy_ready.set()
    _wait_for_run_start(stop_event, run_start)
    if stop_event.is_set():
        return
    camera_source = None
    preview_stop = threading.Event()
    preview_thread = None
    preview_stats = {"frames": 0, "errors": 0}
    episode_observation_thread = None
    episode_observation_stats = {"frames": 0, "errors": 0}
    if config.live_observation:
        camera_source = LiveThreeCameraSource(
            LiveCameraConfig(
                zed_index=config.zed_index,
                zed_eye=config.zed_eye,
                zed_width=config.zed_width,
                zed_height=config.zed_height,
                zed_fps=config.zed_fps,
                realsense_serial_cam2=config.realsense_serial_cam2,
                realsense_serial_cam3=config.realsense_serial_cam3,
                zed_warmup_usable_frames=config.zed_warmup_usable_frames,
                realsense_warmup_usable_frames=config.realsense_warmup_usable_frames,
                min_realsense_mean=config.min_realsense_mean,
            )
        )
        camera_source.open()
        if preview_queue is not None:
            preview_thread = threading.Thread(
                target=_camera_preview_publisher,
                args=(config, stop_event, preview_stop, camera_source, preview_queue, preview_stats),
                name="camera-preview-publisher",
                daemon=True,
            )
            preview_thread.start()
        if episode_observation_queue is not None:
            episode_observation_thread = threading.Thread(
                target=_episode_observation_publisher,
                args=(config, stop_event, preview_stop, camera_source, episode_observation_queue, episode_observation_stats),
                name="episode-observation-publisher",
                daemon=True,
            )
            episode_observation_thread.start()
        samples = None
    else:
        source = LeRobotReplaySource(
            config.dataset_root,
            repo_id=config.repo_id,
            episode_index=config.episode_index,
            video_backend=config.video_backend,
        )
        samples = iter(source)
    chunks = 0
    timing = TimingStats()
    warmup_ms: float | None = None
    latest_state = {"state12": np.zeros(12, dtype=np.float32).tolist(), "gripper_open": 1.0}
    pending_request: dict[str, Any] | None = _make_initial_inference_request()
    try:
        while not stop_event.is_set():
            if config.max_inference_chunks is not None and chunks >= config.max_inference_chunks:
                break
            if pending_request is None:
                pending_request = _get_next_inference_request(request_queue, stop_event)
                if pending_request is None:
                    continue
            request_tick = int(pending_request["request_tick"])
            request_time_ns = int(pending_request["request_time_ns"])
            while True:
                try:
                    latest_state = state_queue.get_nowait()
                except queue.Empty:
                    break
            if config.live_observation:
                if camera_source is None:
                    raise RuntimeError("live camera source is not initialized")
                images = camera_source.read_rgb_uint8()
                obs = {
                    key: torch.from_numpy(rgb).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
                    for key, rgb in images.items()
                }
                state13 = np.concatenate(
                    [
                        np.asarray(latest_state["state12"], dtype=np.float32),
                        np.asarray([float(latest_state.get("gripper_open", 1.0))], dtype=np.float32),
                    ]
                )
                obs["observation.state"] = torch.as_tensor(state13, dtype=torch.float32)
                obs["task"] = config.task
                observation_source = "live_cameras_latest_state"
            else:
                if samples is None:
                    raise RuntimeError("replay sample iterator is not initialized")
                try:
                    sample = next(samples)
                except StopIteration:
                    source = LeRobotReplaySource(
                        config.dataset_root,
                        repo_id=config.repo_id,
                        episode_index=config.episode_index,
                        video_backend=config.video_backend,
                    )
                    samples = iter(source)
                    sample = next(samples)
                obs = {k: v for k, v in sample.items() if k not in {"action", "action_is_pad"}}
                observation_source = "lerobot_replay"
            start_ns = time.monotonic_ns()
            batch = preprocessor(obs)
            device_type = "cuda" if device.startswith("cuda") else "cpu"
            with torch.no_grad(), torch.amp.autocast(device_type, enabled=(device_type == "cuda" and config.use_amp)):
                action_chunk = policy.predict_action_chunk(batch)
            raw_chunk = postprocessor.process_action(action_chunk).detach().cpu().numpy().astype(np.float32)
            end_ns = time.monotonic_ns()
            result_tick = _result_tick_for_request(config, pending_request, end_ns)
            if warmup_ms is None:
                warmup_ms = (end_ns - start_ns) / 1_000_000.0
            if raw_chunk.ndim == 3:
                chunk = raw_chunk[0]
            else:
                chunk = raw_chunk
            if chunk.shape[-1] != 7:
                raise ValueError(f"SmolVLA action chunk must end with dim 7, got {chunk.shape}")
            chunk = chunk.copy()
            previous_gripper = float(latest_state.get("gripper_open", config.initial_gripper_open))
            chunk[:, 6] = _postprocess_gripper_sequence(
                chunk[:, 6],
                previous_gripper=previous_gripper,
                close_threshold=config.gripper_close_threshold,
                open_threshold=config.gripper_open_threshold,
            )
            _drain_put(
                chunk_queue,
                {
                    "chunk_id": chunks,
                    "start_step": request_tick,
                    "request_tick": request_tick,
                    "request_time_ns": request_time_ns,
                    "result_tick": result_tick,
                    "result_time_ns": end_ns,
                    "chunk": chunk,
                    "inference_start_ns": start_ns,
                    "inference_end_ns": end_ns,
                },
            )
            if chunks == 0:
                first_action_chunk_ready.set()
                print(
                    f"[soft_vla] first action chunk ready: backend=smolvla, "
                    f"latency_ms={(end_ns - start_ns) / 1_000_000.0:.3f}, "
                    f"chunk_shape={list(chunk.shape)}",
                    flush=True,
                )
            timing.add_ns(end_ns - start_ns)
            _best_effort_put(
                log_queue,
                {
                    "process": "smolvla_inference",
                    "chunk_id": chunks,
                    "real_policy": True,
                    "live_observation": config.live_observation,
                    "observation_source": observation_source,
                    "chunk_shape": list(chunk.shape),
                    "latency_ms": (end_ns - start_ns) / 1_000_000.0,
                    "inference_latency_ms": (end_ns - request_time_ns) / 1_000_000.0,
                    "request_tick": request_tick,
                    "request_time": request_time_ns / 1_000_000_000.0,
                    "result_tick": result_tick,
                    "result_time": end_ns / 1_000_000_000.0,
                    "bootstrap": bool(pending_request.get("bootstrap", False)),
                },
            )
            chunks += 1
            pending_request = None
            if config.max_inference_chunks is not None and chunks >= config.max_inference_chunks:
                stop_event.set()
                break
    finally:
        preview_stop.set()
        if preview_thread is not None:
            preview_thread.join(timeout=2.0)
        if episode_observation_thread is not None:
            episode_observation_thread.join(timeout=2.0)
        if camera_source is not None:
            camera_source.close()
    summary_queue.put(
        {
            "name": "smolvla_inference",
            "summary": {
                "chunks": chunks,
                "real_policy": True,
                "live_observation": config.live_observation,
                "checkpoint": str(checkpoint),
                "preview_frames": preview_stats["frames"],
                "preview_errors": preview_stats["errors"],
                "episode_observation_frames": episode_observation_stats["frames"],
                "episode_observation_errors": episode_observation_stats["errors"],
                "warmup_ms": warmup_ms,
                "timing": timing.summary(),
            },
        }
    )


def _camera_preview_publisher(
    config: SmolVLAAsyncRuntimeConfig,
    stop_event,
    preview_stop,
    camera_source: LiveThreeCameraSource,
    preview_queue,
    preview_stats: dict[str, int],
) -> None:
    timer = PeriodicTimer(max(0.1, float(config.camera_preview_fps)))
    while not stop_event.is_set() and not preview_stop.is_set():
        try:
            images = camera_source.read_rgb_uint8()
            _drain_put(preview_queue, {"monotonic_ns": time.monotonic_ns(), "images": images})
            preview_stats["frames"] += 1
        except Exception:
            preview_stats["errors"] += 1
            time.sleep(0.05)
            timer.wait_next()


def _episode_observation_publisher(
    config: SmolVLAAsyncRuntimeConfig,
    stop_event,
    preview_stop,
    camera_source: LiveThreeCameraSource,
    episode_observation_queue,
    stats: dict[str, int],
) -> None:
    timer = PeriodicTimer(max(0.1, float(config.upper_frequency_hz)))
    while not stop_event.is_set() and not preview_stop.is_set():
        try:
            images = camera_source.read_rgb_uint8()
            _drain_put(
                episode_observation_queue,
                {
                    "monotonic_ns": time.monotonic_ns(),
                    "timestamp": time.time(),
                    "images": images,
                },
            )
            stats["frames"] += 1
        except Exception:
            stats["errors"] += 1
        finally:
            timer.wait_next()


def camera_preview_process(config: SmolVLAAsyncRuntimeConfig, stop_event, preview_queue, summary_queue) -> None:
    _install_stop_signal_handlers(stop_event)
    shown = 0
    skipped = 0
    saved = 0
    reason = None
    display_available = True
    window_created = False
    fallback_path = None
    if config.log_jsonl:
        fallback_path = Path(config.log_jsonl).parent / "camera_preview_latest.jpg"
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
    import cv2

    window = config.camera_preview_window
    try:
        while not stop_event.is_set():
            try:
                item = preview_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item == STOP:
                break
            canvas = compose_camera_preview(item["images"], scale=config.camera_preview_scale)
            bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            if display_available:
                try:
                    if not window_created:
                        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                        window_created = True
                    cv2.imshow(window, bgr)
                    shown += 1
                    key = cv2.waitKey(1) & 0xFF
                    if key in {27, ord("q")}:
                        stop_event.set()
                        break
                except Exception as exc:
                    display_available = False
                    reason = f"{type(exc).__name__}: {exc}"
            if not display_available and fallback_path is not None:
                cv2.imwrite(str(fallback_path), bgr)
                saved += 1
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
    finally:
        if window_created:
            try:
                cv2.destroyWindow(window)
            except Exception:
                pass
    summary_queue.put(
        {
            "name": "camera_preview",
            "summary": {
                "shown": shown,
                "skipped": skipped,
                "saved": saved,
                "display_available": display_available,
                "fallback_path": None if fallback_path is None else str(fallback_path),
                "reason": reason,
            },
        }
    )


def compose_camera_preview(images: dict[str, np.ndarray], *, scale: float) -> np.ndarray:
    scale = max(0.05, min(float(scale), 1.0))
    cam1 = _resize_for_preview(images["observation.images.cam_1"], scale)
    cam2 = _resize_for_preview(images["observation.images.cam_2"], scale)
    cam3 = _resize_for_preview(images["observation.images.cam_3"], scale)
    target_width = max(cam1.shape[1], cam2.shape[1] + cam3.shape[1])
    top = _pad_to_width(_label_preview(cam1, "cam_1 ZED left"), target_width)
    bottom = np.concatenate([_label_preview(cam2, "cam_2 RealSense"), _label_preview(cam3, "cam_3 RealSense")], axis=1)
    bottom = _pad_to_width(bottom, target_width)
    return np.concatenate([top, bottom], axis=0)


def _resize_for_preview(rgb: np.ndarray, scale: float) -> np.ndarray:
    import cv2

    h, w = rgb.shape[:2]
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    return cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)


def _label_preview(rgb: np.ndarray, label: str) -> np.ndarray:
    import cv2

    out = rgb.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 260), 26), (0, 0, 0), thickness=-1)
    cv2.putText(out, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _pad_to_width(rgb: np.ndarray, width: int) -> np.ndarray:
    if rgb.shape[1] >= width:
        return rgb
    pad = np.zeros((rgb.shape[0], width - rgb.shape[1], 3), dtype=rgb.dtype)
    return np.concatenate([rgb, pad], axis=1)


def logger_process(config: SmolVLAAsyncRuntimeConfig, stop_event, log_queue, summary_queue) -> None:
    _install_stop_signal_handlers(stop_event)
    path = Path(config.log_jsonl) if config.log_jsonl else None
    count = 0
    dropped_stop = False
    fh = path.open("a", encoding="utf-8") if path else None
    try:
        while True:
            try:
                item = log_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item == STOP:
                dropped_stop = True
                break
            count += 1
            if fh:
                fh.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    finally:
        if fh:
            fh.close()
    summary_queue.put({"name": "async_logger", "summary": {"records": count, "saw_stop": dropped_stop, "path": str(path) if path else None}})


def _mock_state12(step: int, frequency_hz: float) -> np.ndarray:
    t = float(step) / float(frequency_hz)
    state = np.zeros(12, dtype=np.float32)
    state[0] = 0.05 + 0.001 * np.sin(t)
    state[1] = 0.65
    state[2] = 0.07
    state[6] = 0.001 * np.cos(t)
    return state


def _best_effort_put(q, item) -> None:
    try:
        q.put_nowait(item)
    except queue.Full:
        pass


def _drain_put(q, item) -> None:
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                return


def _install_stop_signal_handlers(stop_event) -> tuple[dict[int, Any], dict[str, Any]]:
    previous: dict[int, Any] = {}
    state: dict[str, Any] = {"signum": None, "name": None}

    def handler(signum, frame) -> None:  # noqa: ARG001
        state["signum"] = signum
        state["name"] = signal.Signals(signum).name
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[sig] = signal.getsignal(sig)
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass
    return previous, state


def _restore_signal_handlers(previous: dict[int, Any]) -> None:
    for sig, old_handler in previous.items():
        try:
            signal.signal(sig, old_handler)
        except (ValueError, OSError):
            pass


def _send_zero_pressure_safely(driver, *, attempts: int, interval_s: float) -> tuple[int, list[str]]:
    packets = 0
    errors: list[str] = []
    for _ in range(max(1, int(attempts))):
        try:
            driver.send_zero()
            packets += 1
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        time.sleep(max(0.0, float(interval_s)))
    return packets, errors


def _wait_for_motion_policy_ready(config: SmolVLAAsyncRuntimeConfig, stop_event, motion_policy_ready) -> None:
    deadline = time.monotonic() + float(config.motion_policy_ready_timeout_s)
    while not stop_event.is_set():
        if motion_policy_ready.is_set():
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"motion policy was not initialized within {config.motion_policy_ready_timeout_s:.1f}s; "
                "SmolVLA inference will not start"
            )
        time.sleep(0.05)


def _wait_for_run_start(stop_event, run_start) -> None:
    while not stop_event.is_set() and not run_start.is_set():
        time.sleep(0.05)


def _wait_for_first_action_chunk_if_needed(
    config: SmolVLAAsyncRuntimeConfig,
    stop_event,
    first_action_chunk_ready,
    process_name: str,
) -> None:
    if not config.wait_for_first_action_chunk:
        return
    deadline = time.monotonic() + float(config.first_action_timeout_s)
    while not stop_event.is_set():
        if first_action_chunk_ready.is_set():
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"{process_name} timed out waiting for first action chunk after {config.first_action_timeout_s:.1f}s"
            )
        time.sleep(0.05)


def _make_initial_inference_request() -> dict[str, Any]:
    now_ns = time.monotonic_ns()
    return {
        "request_tick": 0,
        "request_time_ns": now_ns,
        "mode": "bootstrap",
        "reason": "initial_first_chunk",
        "bootstrap": True,
    }


def _get_next_inference_request(request_queue, stop_event) -> dict[str, Any] | None:
    while not stop_event.is_set():
        try:
            item = request_queue.get(timeout=0.05)
        except queue.Empty:
            return None
        if item == STOP:
            return None
        return dict(item)
    return None


def _result_tick_for_request(config: SmolVLAAsyncRuntimeConfig, request: dict[str, Any], result_time_ns: int) -> int:
    request_tick = int(request["request_tick"])
    if request.get("bootstrap", False):
        return request_tick
    action_dt_ns = int(round(1_000_000_000.0 / float(config.upper_frequency_hz)))
    elapsed_ns = max(0, int(result_time_ns) - int(request["request_time_ns"]))
    elapsed_ticks = int(np.ceil(float(elapsed_ns) / float(action_dt_ns)))
    return request_tick + max(0, elapsed_ticks)


def _wait_for_ready_events(config: SmolVLAAsyncRuntimeConfig, stop_event, events: list[tuple[str, Any]]) -> None:
    deadline = time.monotonic() + float(config.motion_policy_ready_timeout_s)
    pending = {name: event for name, event in events}
    while pending and not stop_event.is_set():
        for name, event in list(pending.items()):
            if event.is_set():
                pending.pop(name)
        if pending and time.monotonic() >= deadline:
            raise TimeoutError(
                f"not ready before start-key prompt within {config.motion_policy_ready_timeout_s:.1f}s: "
                f"{sorted(pending)}"
            )
        time.sleep(0.05)


def _wait_for_keypress() -> None:
    if not sys.stdin.isatty():
        print("[soft_vla] stdin is not a TTY; continuing without keypress.", flush=True)
        return
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except Exception:
        input()


def _postprocess_gripper_sequence(
    values: np.ndarray,
    *,
    previous_gripper: float,
    close_threshold: float,
    open_threshold: float,
) -> np.ndarray:
    close = float(close_threshold)
    open_ = float(open_threshold)
    if not 0.0 <= close < open_ <= 1.0:
        raise ValueError(f"gripper thresholds must satisfy 0 <= close < open <= 1, got close={close}, open={open_}")
    state = 1.0 if float(previous_gripper) >= 0.5 else 0.0
    out = np.zeros_like(np.asarray(values, dtype=np.float32), dtype=np.float32)
    for i, raw in enumerate(np.asarray(values, dtype=np.float32).reshape(-1)):
        if float(raw) > open_:
            state = 1.0
        elif float(raw) < close:
            state = 0.0
        out[i] = state
    return out


def _format_motion_policy_metadata(metadata: dict[str, Any]) -> str:
    keys = [
        "feedforward",
        "feedback",
        "device",
        "fixed_k_source",
        "fixed_k_path",
        "feedback_gain_scale",
        "max_integral_error",
        "q_tcp6_weight",
        "q_state_tail_weight",
        "q_latent_weight",
        "q_integral_weight",
        "r_weight",
        "pressure_checkpoint",
        "awac_checkpoint",
        "koopman_checkpoint",
    ]
    return ", ".join(f"{key}={metadata.get(key)}" for key in keys if key in metadata)
