from __future__ import annotations

import argparse
import json
import select
import time
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class XboxControllerConfig:
    backend: str = "evdev"
    device_path: str | None = None
    poll_hz: float = 50.0
    print_events: bool = False


class XboxControllerReader:
    """Best-effort Xbox reader.

    The deployment runtime treats controller failures as disconnected snapshots,
    so losing the gamepad does not crash the 10Hz or 50Hz loops.
    """

    def __init__(self, config: XboxControllerConfig | None = None) -> None:
        self.config = config or XboxControllerConfig()
        self._device = None

    def open(self) -> None:
        if self.config.backend != "evdev":
            raise RuntimeError(f"unsupported gamepad backend: {self.config.backend}")
        try:
            import evdev
        except ImportError as exc:
            raise RuntimeError("evdev is required for GAMEPAD_BACKEND=evdev; install python-evdev or use tests/fake input") from exc
        if self.config.device_path:
            self._device = evdev.InputDevice(self.config.device_path)
            return
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        for dev in devices:
            name = (dev.name or "").lower()
            if "xbox" in name or "x-box" in name or "controller" in name or "gamepad" in name:
                self._device = dev
                return
        raise RuntimeError("no Xbox/gamepad evdev device found; run --print-events after confirming /dev/input permissions")

    def close(self) -> None:
        if self._device is not None:
            self._device.close()
            self._device = None

    def event_stream(self) -> Iterable[dict[str, Any]]:
        if self._device is None:
            self.open()
        assert self._device is not None
        for event in self._device.read_loop():
            yield self._event_to_dict(event)

    def _event_to_dict(self, event) -> dict[str, Any]:
        try:
            import evdev
            category = evdev.categorize(event)
            code = getattr(category, "event", event).code
            value = getattr(category, "event", event).value
            name = evdev.ecodes.bytype.get(event.type, {}).get(code, str(code))
        except Exception:
            name = str(getattr(event, "code", "unknown"))
            value = getattr(event, "value", None)
        return {"timestamp": time.time(), "type": int(getattr(event, "type", -1)), "code": name, "value": value}


class EvdevXboxStateDecoder:
    def __init__(self, device) -> None:
        self.device = device
        self.state = empty_gamepad_snapshot(connected=True)

    def update(self, event) -> dict[str, Any]:
        try:
            import evdev
            if event.type == evdev.ecodes.EV_ABS:
                name = evdev.ecodes.ABS.get(event.code, str(event.code))
                value = self._normalize_abs(event.code, event.value)
                axis = {
                    "ABS_X": "left_x",
                    "ABS_Y": "left_y",
                    "ABS_RX": "right_x",
                    "ABS_RY": "right_y",
                    "ABS_Z": "lt",
                    "ABS_RZ": "rt",
                }.get(name)
                if axis is not None:
                    self.state["axes"][axis] = value
            elif event.type == evdev.ecodes.EV_KEY:
                name = evdev.ecodes.bytype.get(evdev.ecodes.EV_KEY, {}).get(event.code, str(event.code))
                button = _button_from_evdev_name(name)
                if button is not None:
                    self.state["buttons"][button] = bool(event.value)
        except Exception:
            pass
        self.state["connected"] = True
        self.state["timestamp"] = time.time()
        return dict(self.state)

    def snapshot(self) -> dict[str, Any]:
        snap = empty_gamepad_snapshot(connected=bool(self.state.get("connected", False)))
        snap["axes"].update(self.state.get("axes", {}))
        snap["buttons"].update(self.state.get("buttons", {}))
        snap["esc"] = bool(self.state.get("esc", False))
        snap["timestamp"] = time.time()
        return snap

    def _normalize_abs(self, code: int, value: int) -> float:
        info = self.device.absinfo(code)
        lo = float(info.min)
        hi = float(info.max)
        if hi <= lo:
            return 0.0
        norm = 2.0 * ((float(value) - lo) / (hi - lo)) - 1.0
        try:
            import evdev
            name = evdev.ecodes.ABS.get(code, str(code))
            if name in {"ABS_Z", "ABS_RZ"}:
                norm = (float(value) - lo) / (hi - lo)
        except Exception:
            pass
        return float(max(-1.0, min(1.0, norm)))


def empty_gamepad_snapshot(*, connected: bool = False) -> dict[str, Any]:
    return {
        "connected": connected,
        "axes": {"left_x": 0.0, "left_y": 0.0, "right_x": 0.0, "right_y": 0.0, "lt": 0.0, "rt": 0.0},
        "buttons": {"a": False, "b": False, "x": False, "y": False},
        "esc": False,
        "timestamp": time.time(),
    }


def _button_from_evdev_name(name: Any) -> str | None:
    names = name if isinstance(name, (list, tuple, set)) else [name]
    aliases = {
        "BTN_SOUTH": "a",
        "BTN_A": "a",
        "BTN_EAST": "b",
        "BTN_B": "b",
        "BTN_WEST": "x",
        "BTN_X": "x",
        "BTN_NORTH": "y",
        "BTN_Y": "y",
    }
    for item in names:
        button = aliases.get(str(item))
        if button is not None:
            return button
    return None


def run_print_events(config: XboxControllerConfig) -> None:
    reader = XboxControllerReader(config)
    deadline = None
    if config.poll_hz > 0:
        # Reuse poll_hz field as a lightweight duration carrier for the CLI
        # debug path; deployment uses it as frequency through the runtime config.
        deadline = time.time() + float(config.poll_hz)
    try:
        reader.open()
        print(json.dumps({"event": "opened", "device": str(reader._device)}, ensure_ascii=False), flush=True)
        while deadline is None or time.time() < deadline:
            assert reader._device is not None
            ready, _, _ = select.select([reader._device.fd], [], [], 0.1)
            if not ready:
                continue
            for event in reader._device.read():
                print(json.dumps(reader._event_to_dict(event), ensure_ascii=False), flush=True)
    finally:
        reader.close()


def list_gamepad_devices() -> list[dict[str, Any]]:
    try:
        import evdev
    except ImportError:
        return [{"error": "evdev is not installed"}]
    out = []
    for path in evdev.list_devices():
        dev = evdev.InputDevice(path)
        out.append({"path": path, "name": dev.name, "phys": dev.phys})
        dev.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Xbox controller debug helper.")
    parser.add_argument("--backend", default="evdev", choices=["evdev"])
    parser.add_argument("--device-path", default=None)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--print-events", action="store_true")
    args = parser.parse_args()
    if args.list_devices:
        print(json.dumps({"devices": list_gamepad_devices()}, ensure_ascii=False, indent=2))
        return
    if not args.print_events:
        raise SystemExit("Use --list-devices or --print-events to inspect detected axis/button names.")
    run_print_events(
        XboxControllerConfig(
            backend=args.backend,
            device_path=args.device_path,
            poll_hz=args.duration_s,
            print_events=True,
        )
    )


if __name__ == "__main__":
    main()
