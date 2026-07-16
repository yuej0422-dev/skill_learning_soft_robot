from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any


class AsyncJsonlLogger:
    def __init__(self, path: str | Path, *, max_queue: int = 10000) -> None:
        self.path = Path(path)
        self.queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=max_queue)
        self.thread: threading.Thread | None = None
        self.dropped = 0

    def start(self) -> None:
        if self.thread is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._run, name="async-jsonl-logger", daemon=True)
        self.thread.start()

    def log(self, record: dict[str, Any]) -> None:
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1

    def close(self, timeout: float = 5.0) -> None:
        if self.thread is None:
            return
        self.queue.put(None)
        self.thread.join(timeout=timeout)
        self.thread = None

    def _run(self) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
                self.queue.task_done()

