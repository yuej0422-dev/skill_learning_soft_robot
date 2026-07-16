from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np


class PressureDriver(Protocol):
    packet_channels: int

    def open(self) -> None: ...

    def send_physical(self, pressure_physical: np.ndarray) -> int: ...

    def send_zero(self) -> int: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class SerialPressureDriverConfig:
    port: str = "COM3"
    baudrate: int = 115200
    packet_channels: int = 16
    open_on_init: bool = False
    write_timeout_s: float | None = 0.02


@dataclass
class MockPressureDriver:
    packet_channels: int = 16
    opened: bool = False
    packets: list[np.ndarray] = field(default_factory=list)

    def open(self) -> None:
        self.opened = True

    def send_physical(self, pressure_physical: np.ndarray) -> int:
        if not self.opened:
            raise RuntimeError("mock pressure driver is not open")
        arr = _select_packet_channels(pressure_physical, self.packet_channels)
        self.packets.append(arr.copy())
        return arr.size * 8

    def send_zero(self) -> int:
        return self.send_physical(np.zeros(self.packet_channels, dtype=np.float32))

    def close(self) -> None:
        self.opened = False


class SerialPressureDriver:
    def __init__(self, config: SerialPressureDriverConfig) -> None:
        if config.packet_channels != 16:
            raise ValueError("serial pressure control requires packet_channels=16")
        self.config = config
        self.packet_channels = int(config.packet_channels)
        self._serial = None
        if config.open_on_init:
            self.open()

    def open(self) -> None:
        if self._serial is not None:
            return
        try:
            import serial
        except ImportError as exc:  # pragma: no cover - depends on deployment env
            raise RuntimeError("pyserial is required for SerialPressureDriver") from exc
        self._serial = serial.Serial(
            self.config.port,
            self.config.baudrate,
            write_timeout=self.config.write_timeout_s,
        )

    def send_physical(self, pressure_physical: np.ndarray) -> int:
        if self._serial is None:
            raise RuntimeError("serial pressure driver is not open")
        arr = _select_packet_channels(pressure_physical, self.packet_channels)
        payload = struct.pack("d" * self.packet_channels, *[float(v) for v in arr])
        return int(self._serial.write(payload))

    def send_zero(self) -> int:
        return self.send_physical(np.zeros(self.packet_channels, dtype=np.float32))

    def close(self) -> None:
        if self._serial is None:
            return
        try:
            self.send_zero()
        finally:
            self._serial.close()
            self._serial = None


def _select_packet_channels(pressure_physical: np.ndarray, packet_channels: int) -> np.ndarray:
    arr = np.asarray(pressure_physical, dtype=np.float64).reshape(-1)
    if packet_channels not in (12, 16):
        raise ValueError("packet_channels must be 12 or 16")
    if arr.size == packet_channels:
        selected = arr
    elif arr.size == 16 and packet_channels == 12:
        selected = arr[:12]
    else:
        raise ValueError(f"cannot send pressure vector of size {arr.size} with packet_channels={packet_channels}")
    if not np.all(np.isfinite(selected)):
        raise ValueError("pressure packet contains NaN or Inf")
    return selected.astype(np.float64, copy=False)


def resolve_default_serial_port() -> str:
    """Return a conservative serial default without probing or opening hardware."""
    candidates = ["/dev/ttyUSB0", "/dev/ttyACM0", "COM3"]
    for item in candidates:
        if item.startswith("/") and Path(item).exists():
            return item
    return "COM3"
