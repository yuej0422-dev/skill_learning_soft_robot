from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soft_vla.schemas import GRIPPER_ACTION_INDEX, validate_action


@dataclass(frozen=True)
class SafetyFilter:
    max_abs_delta_translation_m: float = 0.02
    max_abs_delta_rotation_rad: float = 0.08

    def filter_action(self, action: np.ndarray) -> np.ndarray:
        arr = validate_action(action, require_binary_gripper=False).astype(np.float32).copy()
        arr[:3] = np.clip(arr[:3], -self.max_abs_delta_translation_m, self.max_abs_delta_translation_m)
        arr[3:6] = np.clip(arr[3:6], -self.max_abs_delta_rotation_rad, self.max_abs_delta_rotation_rad)
        arr[GRIPPER_ACTION_INDEX] = 1.0 if arr[GRIPPER_ACTION_INDEX] >= 0.5 else 0.0
        return validate_action(arr)
