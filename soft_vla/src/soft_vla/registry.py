from __future__ import annotations

from soft_vla.hardware.null_controller import NullRobotController


CONTROLLERS = {
    "null": NullRobotController,
}


def get_controller(name: str):
    try:
        return CONTROLLERS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown controller '{name}'. Available: {sorted(CONTROLLERS)}") from exc

