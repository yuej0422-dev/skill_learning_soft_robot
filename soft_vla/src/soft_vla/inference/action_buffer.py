from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ActionBuffer:
    maxlen: int = 50
    _queue: deque[np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        self._queue = deque(maxlen=self.maxlen)

    def extend(self, actions) -> None:
        for action in actions:
            self._queue.append(np.asarray(action, dtype=np.float32))

    def pop(self) -> np.ndarray | None:
        if not self._queue:
            return None
        return self._queue.popleft()

    def __len__(self) -> int:
        return len(self._queue)

