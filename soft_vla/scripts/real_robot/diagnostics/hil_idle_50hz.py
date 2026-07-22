from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import sys as _sys
from pathlib import Path as _Path

_COMPONENTS_DIR = _Path(__file__).resolve().parents[1] / "components"
if str(_COMPONENTS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_COMPONENTS_DIR))

from bootstrap import add_src_to_path

add_src_to_path()

from soft_vla.real_robot.pressure_driver import SerialPressureDriver, SerialPressureDriverConfig
from soft_vla.real_robot.robot_io import LuMoStateSource, LuMoStateSourceConfig
from soft_vla.real_robot.safety_manager import SafetyLimits, SafetyManager
from soft_vla.runtime.async_logger import AsyncJsonlLogger
from soft_vla.runtime.timing import PeriodicTimer, TimingStats


def main() -> None:
    parser = argparse.ArgumentParser(description="50 Hz hardware-in-the-loop idle test: read state and send safe constant pressure.")
    parser.add_argument("--hardware-enabled", action="store_true")
    parser.add_argument("--ip", default="192.168.140.1")
    parser.add_argument("--rigid-body-id", type=int, default=1)
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--packet-channels", type=int, choices=[12, 16], default=16)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--frequency", type=float, default=50.0)
    parser.add_argument("--initial-pressure-norm", type=float, default=0.0)
    parser.add_argument("--log-jsonl", type=Path, default=None)
    args = parser.parse_args()
    if not args.hardware_enabled:
        raise SystemExit("Refusing to open hardware without --hardware-enabled.")
    if args.initial_pressure_norm < 0 or args.initial_pressure_norm > 0.2:
        raise SystemExit("--initial-pressure-norm must be in [0, 0.2] for HIL idle.")

    state_source = LuMoStateSource(LuMoStateSourceConfig(ip=args.ip, rigid_body_id=args.rigid_body_id))
    driver = SerialPressureDriver(
        SerialPressureDriverConfig(port=args.port, baudrate=args.baudrate, packet_channels=args.packet_channels)
    )
    safety = SafetyManager(SafetyLimits(slew_rate_physical_per_s=3.0))
    logger = AsyncJsonlLogger(args.log_jsonl) if args.log_jsonl else None
    timing = TimingStats()
    steps = max(1, int(round(args.duration_s * args.frequency)))
    timer = PeriodicTimer(args.frequency)
    state_source.open()
    driver.open()
    if logger:
        logger.start()
    try:
        for step in range(steps):
            t0 = time.monotonic_ns()
            state = state_source.read_state(blocking=True)
            cmd = safety.build_pressure_command(
                motion_norm12=np.full(12, args.initial_pressure_norm, dtype=np.float32),
                gripper_open=state.gripper_open,
                now_ns=t0,
                state_timestamp_ns=state.monotonic_ns,
                current_state12=state.state12,
            )
            written = driver.send_physical(cmd.final_physical)
            timing.add_ns(time.monotonic_ns() - t0)
            if logger:
                logger.log(
                    {
                        "step": step,
                        "state": state.state13.tolist(),
                        "pressure": cmd.final_physical.tolist(),
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

    print(json.dumps({"steps": steps, "timing": timing.summary()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
