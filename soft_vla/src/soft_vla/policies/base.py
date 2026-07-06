from __future__ import annotations

from typing import Protocol

import numpy as np


class Policy(Protocol):
    def predict_action(self, sample: dict) -> np.ndarray:
        ...

