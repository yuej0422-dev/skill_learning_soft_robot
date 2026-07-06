from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from soft_vla.schemas import validate_action


@dataclass
class NullRobotController:
    dry_run: bool = True
    recorded_actions: list[np.ndarray] = field(default_factory=list)

    def send_action(self, action: np.ndarray) -> None:
        arr = validate_action(action).astype(np.float32)
        self.recorded_actions.append(arr.copy())

    def close(self) -> None:
        return None

