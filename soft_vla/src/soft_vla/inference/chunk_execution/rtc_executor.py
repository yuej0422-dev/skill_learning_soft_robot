from __future__ import annotations

from .base import RTCUnavailableError


def probe_official_rtc() -> dict:
    missing = []
    try:
        import lerobot

        version = getattr(lerobot, "__version__", None)
    except Exception as exc:
        return {"available": False, "lerobot_version": None, "missing": ["lerobot"], "error": repr(exc)}
    try:
        from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: F401
    except Exception as exc:
        missing.append(f"RTCConfig: {exc!r}")
    try:
        from lerobot.policies.rtc.action_queue import ActionQueue  # noqa: F401
    except Exception as exc:
        missing.append(f"ActionQueue: {exc!r}")
    return {"available": not missing, "lerobot_version": version, "missing": missing}


class RTCExecutor:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.compat = probe_official_rtc()
        if not self.compat["available"]:
            raise RTCUnavailableError(f"Official LeRobot RTC API unavailable: {self.compat}")
        raise RTCUnavailableError(
            "Official RTC API is present, but this project has not validated SmolVLA RTC checkpoint "
            "compatibility in the current environment. RTC offline execution is NOT_RUN."
        )

