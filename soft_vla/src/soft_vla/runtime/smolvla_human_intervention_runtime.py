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

from soft_vla.human_intervention.action_mapper import HumanActionMapper, HumanActionMapperConfig
from soft_vla.human_intervention.episode_saver import HumanEpisodeSaver
from soft_vla.human_intervention.intervention_manager import InterventionManager, InterventionManagerConfig
from soft_vla.human_intervention.target_integrator import HumanTargetIntegrator, HumanTargetIntegratorConfig
from soft_vla.human_intervention.xbox_controller import EvdevXboxStateDecoder, XboxControllerConfig, XboxControllerReader, empty_gamepad_snapshot
from soft_vla.inference.chunk_execution.registry import make_chunk_executor
from soft_vla.motion_control.reference_generator import ReferenceGenerator, ReferenceGeneratorConfig
from soft_vla.runtime.shared_state import UpperAction
from soft_vla.runtime.smolvla_async_runtime import (
    STOP,
    SmolVLAAsyncRuntimeConfig,
    camera_preview_process,
    control_process_50hz,
    logger_process,
    smolvla_inference_process,
    _best_effort_put,
    _drain_put,
    _install_stop_signal_handlers,
    _restore_signal_handlers,
    _wait_for_first_action_chunk_if_needed,
    _wait_for_keypress,
    _wait_for_ready_events,
    _wait_for_run_start,
)
from soft_vla.runtime.timing import PeriodicTimer, TimingStats


@dataclass(frozen=True)
class HumanInterventionRuntimeConfig(SmolVLAAsyncRuntimeConfig):
    human_intervention: bool = True
    gamepad_backend: str = "evdev"
    gamepad_device_path: str | None = None
    gamepad_poll_hz: float = 50.0
    print_gamepad_events: bool = False
    joystick_deadzone: float = 0.15
    intervention_release_deadzone: float = 0.10
    intervention_release_ticks: int = 1
    human_max_delta_pos: float = 0.001
    human_max_delta_rot: float = 0.005
    rotation_enabled: bool = False
    rotation_axis: str = "none"
    human_target_integration: bool = True
    human_target_max_pos_offset: float = 0.20
    human_target_max_rot_offset: float = 1.00
    handover_blend_steps: int = 2
    blend_tcp_only: bool = True
    blend_gripper: bool = False
    clear_vla_action_queue_on_intervention: bool = False
    request_fresh_vla_on_release: bool = False
    vla_shadow_mode_during_intervention: bool = True
    seamless_policy_resume: bool = True
    remote_control_debug: bool = False
    save_human_episodes: bool = True
    episode_save_root: str = "/tmp/soft_vla_human_episodes"


def run_smolvla_human_intervention_runtime(config: HumanInterventionRuntimeConfig) -> dict[str, Any]:
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
    human_state_queue: mp.Queue = ctx.Queue(maxsize=8)
    episode_queue: mp.Queue = ctx.Queue(maxsize=4096)
    episode_observation_queue: mp.Queue = ctx.Queue(maxsize=4)
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
            name="soft-vla-human-upper-10hz",
            target=human_upper_dispatch_process_10hz,
            args=(
                config,
                stop_event,
                run_start,
                first_action_chunk_ready,
                reference_queue,
                upper_state_queue,
                inference_request_queue,
                chunk_queue,
                human_state_queue,
                episode_observation_queue,
                episode_queue,
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
                episode_observation_queue,
            ),
        ),
        ctx.Process(name="soft-vla-xbox-listener", target=xbox_listener_process, args=(config, stop_event, human_state_queue, log_queue, summary_queue)),
        ctx.Process(name="soft-vla-human-episode-saver", target=episode_saver_process, args=(config, stop_event, episode_queue, summary_queue)),
        ctx.Process(name="soft-vla-async-logger", target=logger_process, args=(config, stop_event, log_queue, summary_queue)),
    ]
    if preview_queue is not None:
        processes.append(ctx.Process(name="soft-vla-camera-preview", target=camera_preview_process, args=(config, stop_event, preview_queue, summary_queue)))

    previous_signal_handlers, signal_state = _install_stop_signal_handlers(stop_event)
    for process in processes:
        process.start()

    try:
        if config.wait_for_start_key:
            _wait_for_ready_events(config, stop_event, [("motion_policy", motion_policy_ready), ("vla_policy", vla_policy_ready)])
            if not stop_event.is_set():
                print("[soft_vla] human-intervention runtime ready. Press any key to start execution.", flush=True)
                _wait_for_keypress()
                print("[soft_vla] start signal received; entering human-intervention runtime loops.", flush=True)
                run_start.set()
        else:
            run_start.set()
        _wait_for_first_action_chunk_if_needed(config, stop_event, first_action_chunk_ready, "main")
        if float(config.duration_s) <= 0.0:
            while not stop_event.is_set():
                time.sleep(0.05)
        else:
            deadline = time.monotonic() + float(config.duration_s)
            while time.monotonic() < deadline and not stop_event.is_set():
                time.sleep(0.05)
    finally:
        stop_event.set()
        _best_effort_put(log_queue, STOP)
        _best_effort_put(episode_queue, STOP)
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


def human_upper_dispatch_process_10hz(
    config: HumanInterventionRuntimeConfig,
    stop_event,
    run_start,
    first_action_chunk_ready,
    reference_queue,
    state_queue,
    request_queue,
    chunk_queue,
    human_state_queue,
    episode_observation_queue,
    episode_queue,
    log_queue,
    summary_queue,
) -> None:
    _install_stop_signal_handlers(stop_event)
    _wait_for_run_start(stop_event, run_start)
    _wait_for_first_action_chunk_if_needed(config, stop_event, first_action_chunk_ready, "human_upper_10hz")
    if stop_event.is_set():
        return
    timer = PeriodicTimer(config.upper_frequency_hz)
    ref_gen_kwargs: dict[str, Any] = {
        "upper_frequency_hz": config.upper_frequency_hz,
        "control_frequency_hz": config.control_frequency_hz,
        "delta_tcp_scale": config.delta_tcp_scale,
    }
    if config.human_target_integration:
        ref_gen_kwargs["max_delta_tcp"] = (
            config.human_target_max_pos_offset,
            config.human_target_max_pos_offset,
            config.human_target_max_pos_offset,
            config.human_target_max_rot_offset,
            config.human_target_max_rot_offset,
            config.human_target_max_rot_offset,
        )
    ref_gen = ReferenceGenerator(ReferenceGeneratorConfig(**ref_gen_kwargs))
    executor = make_chunk_executor({"mode": config.mode, "chunk_size": config.chunk_size, "execution_horizon": config.execution_horizon, "replan_interval": config.replan_interval, "chunk_trigger_margin": config.chunk_trigger_margin, "chunk_expected_stale_steps": config.chunk_expected_stale_steps})
    mapper = HumanActionMapper(_mapper_config(config))
    manager = InterventionManager(_manager_config(config))
    target_integrator = HumanTargetIntegrator(_target_integrator_config(config))
    latest_state = {"state12": np.zeros(12, dtype=np.float32).tolist(), "gripper_open": 1.0}
    latest_human_snapshot = empty_gamepad_snapshot(connected=False)
    latest_episode_observation: dict[str, Any] | None = None
    steps = 0
    fallbacks = 0
    replan_triggers = 0
    human_steps = 0
    period_timing = TimingStats()
    last_step_start_ns: int | None = None
    inference_running = True
    while not stop_event.is_set():
        t0 = time.monotonic_ns()
        period_ms = None if last_step_start_ns is None else (t0 - last_step_start_ns) / 1_000_000.0
        last_step_start_ns = t0
        if period_ms is not None:
            period_timing.samples_ms.append(period_ms)
        latest_state = _drain_latest(state_queue, latest_state)
        latest_human_snapshot = _drain_latest(human_state_queue, latest_human_snapshot)
        latest_episode_observation = _drain_latest(episode_observation_queue, latest_episode_observation)
        while True:
            try:
                chunk_msg = chunk_queue.get_nowait()
            except queue.Empty:
                break
            executor.submit_chunk(chunk_msg["chunk"], observation_timestamp=float(chunk_msg["start_step"]), inference_start_timestamp=float(chunk_msg["inference_start_ns"]), inference_end_timestamp=float(chunk_msg["inference_end_ns"]), request_tick=int(chunk_msg.get("request_tick", chunk_msg["start_step"])), result_tick=int(chunk_msg.get("result_tick", chunk_msg["start_step"])), next_dispatch_tick=steps, drop_stale_actions=True)
            inference_running = False

        vla_record_source = "exception_fallback"
        vla_fallback = False
        try:
            record = executor.get_action(steps, time.monotonic())
            vla_action7 = record.action.copy()
            vla_record_source = record.source
            vla_fallback = "fallback" in record.source
            if vla_fallback:
                fallbacks += 1
                vla_action7[6] = float(latest_state.get("gripper_open", 1.0))
        except Exception:
            fallbacks += 1
            vla_fallback = True
            vla_action7 = np.zeros(7, dtype=np.float32)
            vla_action7[6] = float(latest_state.get("gripper_open", 1.0))
        if config.remote_control_debug:
            vla_action7 = np.zeros(7, dtype=np.float32)
            vla_action7[6] = 0.5
            vla_record_source = "remote_control_debug_zero_vla"
            vla_fallback = False

        human_cmd = mapper.map_input(latest_human_snapshot, current_state12=np.asarray(latest_state["state12"], dtype=np.float32))
        result = manager.step(vla_action7=vla_action7, human_command=human_cmd, vla_fallback=vla_fallback)
        target_result = target_integrator.step(result.executed_action7, active=result.action_source == "human")
        executed_action7 = target_result.action7
        human_steps += int(result.action_source == "human")
        if result.handover_event == "vla_to_human":
            print("[HUMAN] handover vla -> human", flush=True)
            print("[HUMAN] intervention start", flush=True)
        elif result.handover_event == "human_to_vla":
            print("[HUMAN] handover human -> vla", flush=True)
            print("[HUMAN] seamless resume using current VLA stream", flush=True)
            print("[HUMAN] intervention end", flush=True)
        if human_cmd.gripper_command == 0:
            print("[GRIPPER] A close", flush=True)
        elif human_cmd.gripper_command == 1:
            print("[GRIPPER] Y open", flush=True)

        action = UpperAction(delta_tcp6=executed_action7[:6], gripper_open=float(executed_action7[6]), upper_step=steps, source=f"human_intervention_{result.action_source}")
        segment = ref_gen.build(current_state12=np.asarray(latest_state["state12"], dtype=np.float32), action=action)
        _drain_put(reference_queue, {"upper_step": steps, "control_start_step": segment.control_start_step, "reference_states12": segment.reference_states12.tolist(), "gripper_open": segment.gripper_open, "delta_tcp6": action.delta_tcp6.tolist(), "action_source": result.action_source})

        replan_triggered = False
        if executor.needs_replan(steps, time.monotonic()) and not inference_running:
            try:
                request_queue.put_nowait({"request_tick": steps + 1, "request_time_ns": time.monotonic_ns(), "mode": config.mode, "reason": "executor_needs_replan"})
                inference_running = True
                replan_triggered = True
                replan_triggers += 1
            except queue.Full:
                inference_running = True

        frame = {
            "process": "human_upper_10hz",
            "tick": steps,
            "timestamp": t0 / 1_000_000_000.0,
            "period_ms": period_ms,
            "vla_action": result.vla_action7.astype(float).tolist(),
            "human_action": result.human_action7.astype(float).tolist(),
            "executed_action": executed_action7.astype(float).tolist(),
            "executed_action_delta_tcp": executed_action7[:6].astype(float).tolist(),
            "executed_action_gripper": float(executed_action7[6]),
            "action_source": result.action_source,
            "previous_action_source": result.previous_action_source,
            "intervention_active": result.intervention_active,
            "human_input_norm": result.human_input_norm,
            "gamepad_connected": result.gamepad_connected,
            "gripper_action": human_cmd.gripper_command,
            "success_button_pressed": human_cmd.success_pressed,
            "failure_button_pressed": human_cmd.failure_pressed,
            "reset_triggered": result.reset_triggered,
            "handover_event": result.handover_event,
            "handover_blend_active": result.handover_blend_active,
            "handover_blend_step": result.handover_blend_step,
            "human_target_integration_enabled": config.human_target_integration,
            "human_target_integrated_delta6": target_result.accumulated_delta6.astype(float).tolist(),
            "human_target_xz_direction": target_result.xz_direction,
            "human_target_y_direction": target_result.y_direction,
            "human_target_rot_direction": target_result.rot_direction,
            "human_target_integrator_reset": target_result.reset,
            "vla_shadow_mode_active": True,
            "fallback_used": result.fallback_used,
            "vla_record_source": vla_record_source,
            "replan_triggered": replan_triggered,
            "inference_running": inference_running,
            "state12": latest_state["state12"],
        }
        _best_effort_put(log_queue, frame)
        episode_frame = {
            "tick": steps,
            "timestamp": float(steps) / float(config.upper_frequency_hz),
            "executed_action": executed_action7.astype(float).tolist(),
            "executed_action_delta_tcp": executed_action7[:6].astype(float).tolist(),
            "executed_action_gripper": float(executed_action7[6]),
            "state12": latest_state["state12"],
            "u_p12": latest_state.get("u_p12"),
            "u_paw4": latest_state.get("u_paw4"),
            "motion_norm12": latest_state.get("motion_norm12"),
            "pressure16": latest_state.get("pressure16"),
            "images": None if latest_episode_observation is None else latest_episode_observation.get("images"),
        }
        _best_effort_put(episode_queue, {"event": "frame", "frame": episode_frame})
        if result.termination_reason in {"x_success", "b_failure"}:
            _best_effort_put(episode_queue, {"event": "close_episode", "success": result.termination_reason == "x_success", "failure": result.termination_reason == "b_failure", "termination_reason": result.termination_reason})
            print(f"[EPISODE] {'success' if result.termination_reason == 'x_success' else 'failure'} saved", flush=True)
            _drain_put(
                reference_queue,
                {
                    "event": "episode_end_reset_zero",
                    "reason": result.termination_reason,
                    "hold_s": config.episode_end_reset_sleep_s,
                    "zero_packets": config.episode_end_reset_zero_packets,
                },
            )
            print(
                f"[SAFETY] X/B episode end: reset 16 pressure channels to zero, then sleep {config.episode_end_reset_sleep_s:.1f}s before next episode",
                flush=True,
            )
            time.sleep(max(0.0, float(config.episode_end_reset_sleep_s)) + 0.2)
            while True:
                try:
                    chunk_queue.get_nowait()
                except queue.Empty:
                    break
            mapper.reset(gripper_open=float(executed_action7[6]))
            manager = InterventionManager(_manager_config(config))
            target_integrator.reset()
            executor.reset()
            latest_human_snapshot = empty_gamepad_snapshot(connected=True)
            inference_running = False
            print("[EPISODE] next episode recording started", flush=True)
        elif result.termination_reason == "esc_interrupted":
            _best_effort_put(episode_queue, {"event": "close_episode", "success": False, "failure": False, "termination_reason": "esc_interrupted"})
            print("[SAFETY] keyboard ESC shutdown", flush=True)
            stop_event.set()
        steps += 1
        timer.wait_next()
    summary_queue.put({"name": "human_upper_10hz", "summary": {"steps": steps, "fallbacks": fallbacks, "human_steps": human_steps, "replan_triggers": replan_triggers, "period_timing": period_timing.summary()}})


def xbox_listener_process(config: HumanInterventionRuntimeConfig, stop_event, human_state_queue, log_queue, summary_queue) -> None:
    _install_stop_signal_handlers(stop_event)
    sent = 0
    errors = 0
    timer = PeriodicTimer(max(1.0, float(config.gamepad_poll_hz)))
    snapshot = empty_gamepad_snapshot(connected=False)
    reader = XboxControllerReader(XboxControllerConfig(backend=config.gamepad_backend, device_path=config.gamepad_device_path, poll_hz=config.gamepad_poll_hz))
    decoder = None
    try:
        reader.open()
        decoder = EvdevXboxStateDecoder(reader._device)
        print("[HUMAN] VLA shadow mode active during intervention", flush=True)
    except Exception as exc:
        errors += 1
        _best_effort_put(log_queue, {"process": "xbox_listener", "event": "gamepad_unavailable", "error": repr(exc)})
    try:
        while not stop_event.is_set():
            if decoder is not None and reader._device is not None:
                try:
                    import select

                    ready, _, _ = select.select([reader._device.fd], [], [], 0)
                    if ready:
                        for event in reader._device.read():
                            snapshot = decoder.update(event)
                            if config.print_gamepad_events and int(getattr(event, "type", -1)) != 0:
                                print(
                                    json.dumps(
                                        {
                                            "event": "gamepad_decoded",
                                            "buttons": snapshot.get("buttons", {}),
                                            "axes": snapshot.get("axes", {}),
                                        },
                                        ensure_ascii=False,
                                    ),
                                    flush=True,
                                )
                except Exception as exc:
                    errors += 1
                    snapshot = empty_gamepad_snapshot(connected=False)
                    _best_effort_put(log_queue, {"process": "xbox_listener", "event": "gamepad_read_error", "error": repr(exc)})
            if sys.stdin is not None and sys.stdin.isatty():
                try:
                    import select

                    ready, _, _ = select.select([sys.stdin], [], [], 0)
                    if ready and sys.stdin.read(1) == "\x1b":
                        snapshot["esc"] = True
                except Exception:
                    pass
            _drain_put(human_state_queue, snapshot)
            sent += 1
            timer.wait_next()
    finally:
        reader.close()
    summary_queue.put({"name": "xbox_listener", "summary": {"snapshots": sent, "errors": errors, "backend": config.gamepad_backend}})


def episode_saver_process(config: HumanInterventionRuntimeConfig, stop_event, episode_queue, summary_queue) -> None:
    saver = HumanEpisodeSaver(config.episode_save_root, enabled=config.save_human_episodes, zed_eye=config.zed_eye)
    frames = 0
    closed = 0
    while True:
        try:
            item = episode_queue.get(timeout=0.1)
        except queue.Empty:
            if stop_event.is_set():
                break
            continue
        if item == STOP:
            break
        if item.get("event") == "frame":
            saver.record_frame(item["frame"])
            frames += 1
        elif item.get("event") == "close_episode":
            saver.close_episode(success=bool(item.get("success", False)), failure=bool(item.get("failure", False)), termination_reason=str(item.get("termination_reason", "interrupted")))
            closed += 1
            saver = HumanEpisodeSaver(config.episode_save_root, enabled=config.save_human_episodes, zed_eye=config.zed_eye)
    if frames and closed == 0:
        saver.close_episode(success=False, failure=False, termination_reason="interrupted")
    summary_queue.put({"name": "human_episode_saver", "summary": {"frames": frames, "closed_episodes": closed, "root": config.episode_save_root}})


def _mapper_config(config: HumanInterventionRuntimeConfig) -> HumanActionMapperConfig:
    return HumanActionMapperConfig(
        joystick_deadzone=config.joystick_deadzone,
        intervention_release_deadzone=config.intervention_release_deadzone,
        max_delta_pos_per_tick=(config.human_max_delta_pos,) * 3,
        max_delta_rot_per_tick=(config.human_max_delta_rot,) * 3,
        rotation_enabled=config.rotation_enabled,
        rotation_axis=config.rotation_axis,
        max_action_slew_pos=config.human_max_delta_pos,
        max_action_slew_rot=config.human_max_delta_rot,
    )


def _manager_config(config: HumanInterventionRuntimeConfig) -> InterventionManagerConfig:
    return InterventionManagerConfig(
        release_ticks=config.intervention_release_ticks,
        handover_blend_steps=config.handover_blend_steps,
        blend_tcp_only=config.blend_tcp_only,
        blend_gripper=config.blend_gripper,
        clear_vla_action_queue_on_intervention=config.clear_vla_action_queue_on_intervention,
        request_fresh_vla_on_release=config.request_fresh_vla_on_release,
        vla_shadow_mode_during_intervention=config.vla_shadow_mode_during_intervention,
        seamless_policy_resume=config.seamless_policy_resume,
    )


def _target_integrator_config(config: HumanInterventionRuntimeConfig) -> HumanTargetIntegratorConfig:
    return HumanTargetIntegratorConfig(
        enabled=config.human_target_integration,
        max_pos_offset=config.human_target_max_pos_offset,
        max_rot_offset=config.human_target_max_rot_offset,
    )


def _drain_latest(q, default):
    latest = default
    while True:
        try:
            latest = q.get_nowait()
        except queue.Empty:
            return latest
