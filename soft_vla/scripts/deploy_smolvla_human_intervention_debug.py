from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import select
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import add_src_to_path

PROJECT_ROOT = add_src_to_path()
REPO_ROOT = PROJECT_ROOT.parent

from soft_vla.human_intervention.action_mapper import HumanActionMapper, HumanActionMapperConfig  # noqa: E402
from soft_vla.human_intervention.intervention_manager import InterventionManager, InterventionManagerConfig  # noqa: E402
from soft_vla.human_intervention.target_integrator import HumanTargetIntegrator, HumanTargetIntegratorConfig  # noqa: E402
from soft_vla.human_intervention.xbox_controller import (  # noqa: E402
    EvdevXboxStateDecoder,
    XboxControllerConfig,
    XboxControllerReader,
    empty_gamepad_snapshot,
    list_gamepad_devices,
)


STOP = {"event": "stop"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Debug the Xbox human-intervention chain without loading SmolVLA, "
            "opening cameras, or writing robot pressure commands."
        )
    )
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--upper-hz", type=float, default=10.0)
    parser.add_argument("--gamepad-backend", default="evdev", choices=["evdev"])
    parser.add_argument("--gamepad-device-path", default=None)
    parser.add_argument("--gamepad-poll-hz", type=float, default=50.0)
    parser.add_argument("--gamepad-deadzone", type=float, default=0.15)
    parser.add_argument("--intervention-release-deadzone", type=float, default=0.10)
    parser.add_argument("--intervention-release-ticks", type=int, default=1)
    parser.add_argument("--human-max-delta-pos", type=float, default=0.005)
    parser.add_argument("--human-max-delta-rot", type=float, default=0.025)
    parser.add_argument("--rotation-enabled", action="store_true")
    parser.add_argument("--rotation-axis", choices=["none", "roll", "pitch", "yaw", "pitch_yaw"], default="pitch_yaw")
    parser.add_argument("--no-human-target-integration", action="store_true")
    parser.add_argument("--human-target-max-pos-offset", type=float, default=0.01)
    parser.add_argument("--human-target-max-rot-offset", type=float, default=0.05)
    parser.add_argument("--handover-blend-steps", type=int, default=2)
    parser.add_argument("--blend-tcp-only", action="store_true")
    parser.add_argument("--blend-gripper", action="store_true")
    parser.add_argument("--print-all", action="store_true")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        print(json.dumps({"devices": list_gamepad_devices()}, ensure_ascii=False, indent=2))
        return

    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()
    human_state_queue: mp.Queue = ctx.Queue(maxsize=16)
    summary_queue: mp.Queue = ctx.Queue()
    processes = [
        ctx.Process(
            name="soft-vla-debug-xbox-listener",
            target=_debug_xbox_listener_process,
            args=(args, stop_event, human_state_queue, summary_queue),
        ),
        ctx.Process(
            name="soft-vla-debug-human-upper-10hz",
            target=_debug_upper_process,
            args=(args, stop_event, human_state_queue, summary_queue),
        ),
    ]

    signal_state: dict[str, str | None] = {"name": None}

    def _request_stop(signum, _frame) -> None:
        signal_state["name"] = signal.Signals(signum).name
        stop_event.set()

    old_int = signal.signal(signal.SIGINT, _request_stop)
    old_term = signal.signal(signal.SIGTERM, _request_stop)
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + float(args.duration_s)
        while time.monotonic() < deadline and not stop_event.is_set():
            time.sleep(0.05)
    finally:
        stop_event.set()
        _drain_put(human_state_queue, STOP)
        for process in processes:
            process.join(timeout=3.0)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)

    summaries: dict[str, Any] = {}
    while True:
        try:
            item = summary_queue.get_nowait()
        except queue.Empty:
            break
        summaries[str(item.get("name", "unknown"))] = item.get("summary", {})

    report = {
        "ok": all(process.exitcode == 0 for process in processes),
        "interrupted_by": signal_state["name"],
        "exitcodes": {process.name: process.exitcode for process in processes},
        "summaries": summaries,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def _debug_xbox_listener_process(args, stop_event, human_state_queue, summary_queue) -> None:
    _install_child_signal_handlers(stop_event)
    reader = XboxControllerReader(
        XboxControllerConfig(
            backend=args.gamepad_backend,
            device_path=args.gamepad_device_path,
            poll_hz=args.gamepad_poll_hz,
        )
    )
    snapshot = empty_gamepad_snapshot(connected=False)
    sent = 0
    raw_events = 0
    key_events = 0
    errors = 0
    decoder = None
    period_s = 1.0 / max(1.0, float(args.gamepad_poll_hz))
    next_tick = time.monotonic()
    try:
        try:
            reader.open()
            decoder = EvdevXboxStateDecoder(reader._device)
            print(
                json.dumps(
                    {
                        "process": "debug_listener",
                        "event": "opened",
                        "device": str(reader._device),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            errors += 1
            print(
                json.dumps(
                    {
                        "process": "debug_listener",
                        "event": "open_error",
                        "error": repr(exc),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        while not stop_event.is_set():
            if decoder is not None and reader._device is not None:
                try:
                    ready, _, _ = select.select([reader._device.fd], [], [], 0)
                    if ready:
                        for event in reader._device.read():
                            raw_events += 1
                            raw = reader._event_to_dict(event)
                            snapshot = decoder.update(event)
                            is_key_event = int(getattr(event, "type", -1)) == 1
                            key_events += int(is_key_event)
                            if args.print_all or is_key_event:
                                print(
                                    json.dumps(
                                        {
                                            "process": "debug_listener",
                                            "event": "raw_decoded",
                                            "raw": raw,
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
                    print(
                        json.dumps(
                            {
                                "process": "debug_listener",
                                "event": "read_error",
                                "error": repr(exc),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

            now = time.monotonic()
            if now >= next_tick:
                if decoder is not None:
                    snapshot = decoder.snapshot()
                _drain_put(human_state_queue, snapshot)
                sent += 1
                next_tick = now + period_s
            time.sleep(0.001)
    finally:
        reader.close()
        summary_queue.put(
            {
                "name": "debug_listener",
                "summary": {
                    "snapshots_sent": sent,
                    "raw_events": raw_events,
                    "key_events": key_events,
                    "errors": errors,
                    "backend": args.gamepad_backend,
                    "device_path": args.gamepad_device_path,
                },
            }
        )


def _debug_upper_process(args, stop_event, human_state_queue, summary_queue) -> None:
    _install_child_signal_handlers(stop_event)
    mapper = HumanActionMapper(
        HumanActionMapperConfig(
            joystick_deadzone=args.gamepad_deadzone,
            intervention_release_deadzone=args.intervention_release_deadzone,
            max_delta_pos_per_tick=(args.human_max_delta_pos,) * 3,
            max_delta_rot_per_tick=(args.human_max_delta_rot,) * 3,
            rotation_enabled=bool(args.rotation_enabled),
            rotation_axis=args.rotation_axis,
            max_action_slew_pos=args.human_max_delta_pos,
            max_action_slew_rot=args.human_max_delta_rot,
        )
    )
    manager = InterventionManager(
        InterventionManagerConfig(
            release_ticks=args.intervention_release_ticks,
            handover_blend_steps=args.handover_blend_steps,
            blend_tcp_only=bool(args.blend_tcp_only),
            blend_gripper=bool(args.blend_gripper),
        )
    )
    target_integrator = HumanTargetIntegrator(
        HumanTargetIntegratorConfig(
            enabled=not bool(args.no_human_target_integration),
            max_pos_offset=args.human_target_max_pos_offset,
            max_rot_offset=args.human_target_max_rot_offset,
        )
    )
    latest_snapshot = empty_gamepad_snapshot(connected=False)
    latest_state12 = np.zeros(12, dtype=np.float32)
    neutral_vla = np.zeros(7, dtype=np.float32)
    neutral_vla[6] = 0.5
    tick = 0
    received = 0
    human_ticks = 0
    last_print_key: tuple[Any, ...] | None = None
    period_s = 1.0 / max(1.0, float(args.upper_hz))
    next_tick = time.monotonic()

    print(
        json.dumps(
            {
                "process": "debug_upper",
                "event": "ready",
                "note": "SmolVLA disabled; pressure output disabled; VLA action is neutral [0,0,0,0,0,0,0.5].",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    while not stop_event.is_set():
        now = time.monotonic()
        if now < next_tick:
            time.sleep(min(0.002, next_tick - now))
            continue
        next_tick += period_s

        while True:
            try:
                item = human_state_queue.get_nowait()
            except queue.Empty:
                break
            if item == STOP:
                stop_event.set()
                break
            latest_snapshot = item
            received += 1

        human_cmd = mapper.map_input(latest_snapshot, current_state12=latest_state12)
        result = manager.step(vla_action7=neutral_vla, human_command=human_cmd, vla_fallback=False)
        target_result = target_integrator.step(result.executed_action7, active=result.action_source == "human")
        executed_action7 = target_result.action7
        human_ticks += int(result.action_source == "human")

        buttons = latest_snapshot.get("buttons", {}) or {}
        axes = latest_snapshot.get("axes", {}) or {}
        print_key = (
            tuple(sorted((k, bool(v)) for k, v in buttons.items())),
            result.action_source,
            human_cmd.gripper_command,
            human_cmd.success_pressed,
            human_cmd.failure_pressed,
            tuple(round(float(executed_action7[i]), 6) for i in range(7)),
        )
        should_print = bool(args.print_all) or print_key != last_print_key
        if should_print:
            print(
                json.dumps(
                    {
                        "process": "debug_upper",
                        "tick": tick,
                        "buttons": buttons,
                        "axes": axes,
                        "human_active": human_cmd.active,
                        "human_input_norm": human_cmd.input_norm,
                        "gripper_command": human_cmd.gripper_command,
                        "success_pressed": human_cmd.success_pressed,
                        "failure_pressed": human_cmd.failure_pressed,
                        "action_source": result.action_source,
                        "handover_event": result.handover_event,
                        "human_action": human_cmd.action7.astype(float).tolist(),
                        "executed_action": executed_action7.astype(float).tolist(),
                        "integrated_delta6": target_result.accumulated_delta6.astype(float).tolist(),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            last_print_key = print_key
        tick += 1

    summary_queue.put(
        {
            "name": "debug_upper",
            "summary": {
                "ticks": tick,
                "snapshots_received": received,
                "human_ticks": human_ticks,
            },
        }
    )


def _drain_put(q, item: Any) -> None:
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                return


def _install_child_signal_handlers(stop_event) -> None:
    def _request_stop(_signum, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)


if __name__ == "__main__":
    main()
