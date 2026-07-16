from __future__ import annotations

import importlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from soft_vla.runtime.shared_state import RobotState


def _import_lumo_sdk_client():
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    sdk_dir = Path(__file__).resolve().parents[1] / "vendor" / "lumo_sdk"
    if str(sdk_dir) not in sys.path:
        sys.path.insert(0, str(sdk_dir))
    try:
        return importlib.import_module("LuMoSDKClient")
    except ImportError as exc:  # pragma: no cover - depends on hardware env
        raise RuntimeError("LuMoSDKClient is required for LuMoStateSource") from exc


class RobotStateSource(Protocol):
    def open(self) -> None: ...

    def read_state(self, *, blocking: bool = True) -> RobotState: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class LuMoStateSourceConfig:
    ip: str = "192.168.140.1"
    rigid_body_id: int = 1
    receive_timeout_ms: int | None = 1000
    connect_on_init: bool = False


@dataclass
class MockRobotStateSource:
    state12: np.ndarray
    gripper_open: float = 0.0
    opened: bool = False

    def open(self) -> None:
        self.opened = True

    def read_state(self, *, blocking: bool = True) -> RobotState:
        if not self.opened:
            raise RuntimeError("mock robot state source is not open")
        return RobotState(
            state12=np.asarray(self.state12, dtype=np.float32),
            gripper_open=self.gripper_open,
            monotonic_ns=time.monotonic_ns(),
            source="mock",
        )

    def close(self) -> None:
        self.opened = False


class LuMoStateSource:
    def __init__(self, config: LuMoStateSourceConfig) -> None:
        self.config = config
        self._client = None
        if config.connect_on_init:
            self.open()

    def open(self) -> None:
        if self._client is not None:
            return
        LuMoSDKClient = _import_lumo_sdk_client()
        LuMoSDKClient.Init()
        LuMoSDKClient.Connnect(self.config.ip)
        self._client = LuMoSDKClient

    def read_state(self, *, blocking: bool = True) -> RobotState:
        if self._client is None:
            raise RuntimeError("LuMo state source is not open")
        if blocking and self.config.receive_timeout_ms is not None:
            deadline = time.monotonic() + float(self.config.receive_timeout_ms) / 1000.0
            frame = None
            while frame is None and time.monotonic() < deadline:
                frame = self._client.ReceiveData(1)
                if frame is None:
                    time.sleep(0.001)
        else:
            frame = self._client.ReceiveData(0 if blocking else 1)
        if frame is None:
            raise TimeoutError("LuMo ReceiveData returned no frame")
        rigid_body_id = int(self.config.rigid_body_id)
        for rigid in frame.rigidBodys:
            if int(rigid.Id) != rigid_body_id:
                continue
            state = np.asarray(
                [
                    0.001 * float(rigid.X),
                    0.001 * float(rigid.Y),
                    0.001 * float(rigid.Z),
                    np.pi * float(rigid.eulerAngle.X) / 180.0,
                    np.pi * float(rigid.eulerAngle.Y) / 180.0,
                    np.pi * float(rigid.eulerAngle.Z) / 180.0,
                    float(rigid.speeds.XfSpeed),
                    float(rigid.speeds.YfSpeed),
                    float(rigid.speeds.ZfSpeed),
                    float(rigid.palstance.fXPalstance),
                    float(rigid.palstance.fYPalstance),
                    float(rigid.palstance.fZPalstance),
                ],
                dtype=np.float32,
            )
            return RobotState(state12=state, monotonic_ns=time.monotonic_ns(), source="lumo")
        raise LookupError(f"rigid body id {rigid_body_id} not found in LuMo frame")

    def close(self) -> None:
        if self._client is not None and hasattr(self._client, "Close"):
            self._client.Close()
        self._client = None
