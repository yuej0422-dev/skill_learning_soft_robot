from __future__ import annotations

from typing import Protocol


class ObservationSource(Protocol):
    def __iter__(self):
        ...

