from __future__ import annotations

from typing import Protocol

import numpy as np


class RobotController(Protocol):
    def send_action(self, action: np.ndarray) -> None:
        ...

    def close(self) -> None:
        ...

