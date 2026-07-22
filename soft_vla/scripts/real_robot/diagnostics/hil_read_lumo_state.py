from __future__ import annotations

import argparse
import json
import time

import numpy as np

import sys as _sys
from pathlib import Path as _Path

_COMPONENTS_DIR = _Path(__file__).resolve().parents[1] / "components"
if str(_COMPONENTS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_COMPONENTS_DIR))

from bootstrap import add_src_to_path

add_src_to_path()

from soft_vla.real_robot.robot_io import LuMoStateSource, LuMoStateSourceConfig
from soft_vla.runtime.timing import PeriodicTimer, TimingStats


def main() -> None:
    parser = argparse.ArgumentParser(description="Read LuMo rigid-body state only. Does not open pressure serial.")
    parser.add_argument("--hardware-enabled", action="store_true")
    parser.add_argument("--ip", default="192.168.140.1")
    parser.add_argument("--rigid-body-id", type=int, default=1)
    parser.add_argument("--receive-timeout-ms", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--frequency", type=float, default=50.0)
    args = parser.parse_args()
    if not args.hardware_enabled:
        raise SystemExit("Refusing to connect LuMo without --hardware-enabled.")

    source = LuMoStateSource(
        LuMoStateSourceConfig(
            ip=args.ip,
            rigid_body_id=args.rigid_body_id,
            receive_timeout_ms=args.receive_timeout_ms,
        )
    )
    timing = TimingStats()
    states: list[list[float]] = []
    source.open()
    timer = PeriodicTimer(args.frequency)
    try:
        for _ in range(args.samples):
            t0 = time.monotonic_ns()
            try:
                state = source.read_state(blocking=True)
            except TimeoutError as exc:
                raise SystemExit(
                    "No LuMo/FZMotion frame received within "
                    f"{args.receive_timeout_ms} ms from {args.ip}:6868. "
                    "Check that the Ubuntu Ethernet IP is in the same subnet, "
                    "the Windows/FZMotion host IP matches --ip, realtime rigid-body "
                    "streaming is enabled, and Windows firewall allows TCP 6868."
                ) from exc
            timing.add_ns(time.monotonic_ns() - t0)
            states.append(state.state12.astype(float).tolist())
            timer.wait_next()
    finally:
        source.close()

    arr = np.asarray(states, dtype=np.float64)
    report = {
        "samples": len(states),
        "timing": timing.summary(),
        "state_mean": arr.mean(axis=0).tolist() if len(states) else [],
        "state_min": arr.min(axis=0).tolist() if len(states) else [],
        "state_max": arr.max(axis=0).tolist() if len(states) else [],
        "finite": bool(np.all(np.isfinite(arr))) if len(states) else False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
