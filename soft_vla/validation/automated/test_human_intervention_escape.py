from __future__ import annotations

import os
import pty
import queue
import sys
import threading
import time

from soft_vla.runtime.smolvla_human_intervention_runtime import (
    HumanInterventionRuntimeConfig,
    _wait_for_runtime_stop_or_terminal_escape,
)


def test_terminal_escape_stops_runtime_and_closes_episode() -> None:
    master_fd, slave_fd = pty.openpty()
    original_stdin = sys.stdin
    tty_stdin = os.fdopen(os.dup(slave_fd), "r", encoding="utf-8")
    stop_event = threading.Event()
    episode_queue: queue.Queue[dict] = queue.Queue()
    worker = threading.Thread(
        target=_wait_for_runtime_stop_or_terminal_escape,
        args=(HumanInterventionRuntimeConfig(duration_s=0.0), stop_event, episode_queue),
        daemon=True,
    )
    try:
        sys.stdin = tty_stdin
        worker.start()
        time.sleep(0.05)
        os.write(master_fd, b"\x1b")
        worker.join(timeout=1.0)

        assert not worker.is_alive()
        assert stop_event.is_set()
        assert episode_queue.get_nowait()["termination_reason"] == "esc_interrupted"
    finally:
        stop_event.set()
        worker.join(timeout=1.0)
        sys.stdin = original_stdin
        tty_stdin.close()
        os.close(master_fd)
        os.close(slave_fd)
