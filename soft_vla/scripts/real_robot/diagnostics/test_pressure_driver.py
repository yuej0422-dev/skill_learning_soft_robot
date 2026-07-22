from __future__ import annotations

import argparse
import json

import numpy as np

import sys as _sys
from pathlib import Path as _Path

_COMPONENTS_DIR = _Path(__file__).resolve().parents[1] / "components"
if str(_COMPONENTS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_COMPONENTS_DIR))

from bootstrap import add_src_to_path

add_src_to_path()

from soft_vla.real_robot.pressure_driver import MockPressureDriver, SerialPressureDriver, SerialPressureDriverConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Test pressure driver. Defaults to mock; real serial requires --real.")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--packet-channels", type=int, choices=[12, 16], default=16)
    parser.add_argument("--pressure", type=float, default=0.0)
    args = parser.parse_args()
    driver = (
        SerialPressureDriver(SerialPressureDriverConfig(args.port, args.baudrate, args.packet_channels))
        if args.real
        else MockPressureDriver(packet_channels=args.packet_channels)
    )
    driver.open()
    try:
        packet = np.full(16, float(args.pressure), dtype=np.float32)
        written = driver.send_physical(packet)
        result = {"mode": "real" if args.real else "mock", "packet_channels": args.packet_channels, "written_bytes": written}
        if isinstance(driver, MockPressureDriver):
            result["last_packet"] = driver.packets[-1].tolist()
        print(json.dumps(result, indent=2))
    finally:
        driver.close()


if __name__ == "__main__":
    main()
