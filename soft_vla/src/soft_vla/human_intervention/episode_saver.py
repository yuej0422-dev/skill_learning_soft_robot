from __future__ import annotations

import csv
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class EpisodeStats:
    episode_id: int
    frame_idx: int = 0
    vla_steps: int = 0
    human_steps: int = 0
    blend_steps: int = 0
    fallback_steps: int = 0
    intervention_ticks: int = 0
    num_intervention_segments: int = 0
    handover_vla_to_human_count: int = 0
    handover_human_to_vla_count: int = 0
    intervention_active_prev: bool = False
    started_at: float = field(default_factory=time.time)


class HumanEpisodeSaver:
    """Episode saver matching the original 3-camera collection layout.

    Differences from ``vla_collect_3cams_v4.py``:
    - no depth folders/columns are written;
    - each CSV row stores the actually executed deployment action.
    """

    def __init__(self, root: str | Path, *, enabled: bool = True, zed_eye: str = "left") -> None:
        self.root = Path(root)
        self.enabled = bool(enabled)
        self.zed_eye = zed_eye
        self.root.mkdir(parents=True, exist_ok=True)
        self.stats = EpisodeStats(episode_id=self._next_episode_id())
        self.episode_dir = self.root / f"episode_{self.stats.episode_id:04d}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.cam1_dir = self.episode_dir / f"images_cam1_zed_{self.zed_eye}"
        self.cam2_dir = self.episode_dir / "images_cam2"
        self.cam3_dir = self.episode_dir / "images_cam3"
        for path in (self.cam1_dir, self.cam2_dir, self.cam3_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.episode_dir / "data.csv"
        self._csv_fh = self.csv_path.open("w", newline="", encoding="utf-8") if self.enabled else None
        self._csv_writer = csv.writer(self._csv_fh) if self._csv_fh is not None else None
        self._image_writer = AsyncImageWriter(enabled=self.enabled)
        if self._csv_writer is not None:
            self._csv_writer.writerow(self._csv_header())

    def _csv_header(self) -> list[str]:
        return _csv_header(self.zed_eye)

    def _csv_action_values(self, row: dict[str, Any]) -> np.ndarray | None:
        action = np.asarray(row.get("executed_action", np.zeros(7, dtype=np.float32)), dtype=np.float64).reshape(-1)
        return action if action.shape == (7,) else None

    def _csv_context_values(self, row: dict[str, Any]) -> list[Any]:
        return []

    def record_frame(self, row: dict[str, Any]) -> None:
        if not self.enabled:
            return
        row = dict(row)
        row.setdefault("episode_id", self.stats.episode_id)
        row.setdefault("frame_idx", self.stats.frame_idx)
        source = row.get("action_source")
        self.stats.vla_steps += int(source == "vla")
        self.stats.human_steps += int(source == "human")
        self.stats.blend_steps += int(bool(row.get("handover_blend_active", False)))
        self.stats.fallback_steps += int(source == "fallback")
        active = bool(row.get("intervention_active", False))
        self.stats.intervention_ticks += int(active)
        if active and not self.stats.intervention_active_prev:
            self.stats.num_intervention_segments += 1
        self.stats.intervention_active_prev = active
        event = row.get("handover_event")
        self.stats.handover_vla_to_human_count += int(event == "vla_to_human")
        self.stats.handover_human_to_vla_count += int(event == "human_to_vla")
        images = row.get("images")
        if not isinstance(images, dict):
            return
        state12 = np.asarray(row.get("state12", np.zeros(12, dtype=np.float32)), dtype=np.float64).reshape(-1)
        action_values = self._csv_action_values(row)
        u_p12 = np.asarray(row.get("u_p12", row.get("motion_norm12", np.zeros(12, dtype=np.float32))), dtype=np.float64).reshape(-1)
        u_paw4 = np.asarray(
            row.get("u_paw4", np.asarray(row.get("pressure16", np.zeros(16, dtype=np.float32)), dtype=np.float64).reshape(-1)[12:16]),
            dtype=np.float64,
        ).reshape(-1)
        if state12.shape[0] < 12 or action_values is None:
            return
        if u_p12.shape[0] != 12:
            u_p12 = np.zeros(12, dtype=np.float64)
        if u_paw4.shape[0] != 4:
            u_paw4 = np.zeros(4, dtype=np.float64)
        img_name = f"{self.stats.frame_idx:06d}.jpg"
        self._write_rgb_image(self.cam1_dir / img_name, images.get("observation.images.cam_1"))
        self._write_rgb_image(self.cam2_dir / img_name, images.get("observation.images.cam_2"))
        self._write_rgb_image(self.cam3_dir / img_name, images.get("observation.images.cam_3"))
        if self._csv_writer is not None:
            timestamp = row.get("timestamp")
            if timestamp is None:
                timestamp = time.time() - self.stats.started_at
            self._csv_writer.writerow(
                [
                    float(timestamp),
                    img_name,
                    img_name,
                    img_name,
                    *self._csv_context_values(row),
                    *u_p12.astype(float).tolist(),
                    *u_paw4.astype(float).tolist(),
                    *action_values.astype(float).tolist(),
                    *state12[:12].astype(float).tolist(),
                ]
            )
            if self.stats.frame_idx % 10 == 0 and self._csv_fh is not None:
                self._csv_fh.flush()
        self.stats.frame_idx += 1

    def close_episode(self, *, success: bool = False, failure: bool = False, termination_reason: str = "interrupted") -> Path:
        self._image_writer.close()
        if self._csv_fh is not None:
            self._csv_fh.close()
            self._csv_fh = None
            self._csv_writer = None
        total = max(1, self.stats.frame_idx)
        meta = {
            "episode_id": self.stats.episode_id,
            "success": bool(success),
            "failure": bool(failure),
            "reason": termination_reason,
            "frame_count": self.stats.frame_idx,
            "vla_steps": self.stats.vla_steps,
            "human_steps": self.stats.human_steps,
            "blend_steps": self.stats.blend_steps,
            "fallback_steps": self.stats.fallback_steps,
            "intervention_ticks": self.stats.intervention_ticks,
            "num_intervention_segments": self.stats.num_intervention_segments,
            "human_takeover_ratio": float(self.stats.human_steps) / float(total),
            "handover_vla_to_human_count": self.stats.handover_vla_to_human_count,
            "handover_human_to_vla_count": self.stats.handover_human_to_vla_count,
            "duration_sec": time.time() - self.stats.started_at,
            "zed_eye": self.zed_eye,
            "sampling_interval_sec": 0.1,
            "data_csv": str(self.csv_path),
            "images_cam1": str(self.cam1_dir),
            "images_cam2": str(self.cam2_dir),
            "images_cam3": str(self.cam3_dir),
            "image_write_errors": self._image_writer.error_count(),
        }
        meta_path = self.episode_dir / "episode_meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta_path

    def _write_rgb_image(self, path: Path, image: Any) -> None:
        if image is None:
            return
        rgb = np.asarray(image)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            return
        bgr = cv2.cvtColor(rgb.astype(np.uint8, copy=False), cv2.COLOR_RGB2BGR)
        self._image_writer.write(path, bgr)

    def _next_episode_id(self) -> int:
        ids = []
        for path in self.root.glob("episode_*"):
            try:
                ids.append(int(path.name.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
        return max(ids) + 1 if ids else 0


class PressureStateHumanEpisodeSaver(HumanEpisodeSaver):
    """Store measured state plus executed and untouched VLA 19D actions.

    No derived transition sidecar is written.  Realized TCP and pressure
    changes can be reconstructed later from adjacent rows in ``data.csv``.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        enabled: bool = True,
        zed_eye: str = "left",
        pressure_delta_scale: float = 1.0,
    ) -> None:
        if float(pressure_delta_scale) <= 0.0:
            raise ValueError("pressure_delta_scale must be positive when saving pressure-state training data")
        super().__init__(root, enabled=enabled, zed_eye=zed_eye)

    def _csv_header(self) -> list[str]:
        return _pressure_state_csv_header(self.zed_eye)

    def _csv_action_values(self, row: dict[str, Any]) -> np.ndarray | None:
        executed = np.asarray(row.get("executed_action"), dtype=np.float64).reshape(-1)
        vla = np.asarray(row.get("vla_action19"), dtype=np.float64).reshape(-1)
        if executed.shape != (19,) or vla.shape != (19,):
            return None
        return np.concatenate([executed, vla])

    def _csv_context_values(self, row: dict[str, Any]) -> list[Any]:
        return [
            str(row.get("action_source", "unknown")),
            int(bool(row.get("intervention_active", False))),
        ]

    def close_episode(
        self,
        *,
        success: bool = False,
        failure: bool = False,
        termination_reason: str = "interrupted",
    ) -> Path:
        meta_path = super().close_episode(
            success=success,
            failure=failure,
            termination_reason=termination_reason,
        )
        if self.enabled:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta.update(
                {
                    "data_csv_action_schema": "source_flags_then_executed_action19_then_raw_vla_action19",
                    "executed_action_dim": 19,
                    "vla_action_dim": 19,
                    "vla_gripper_semantics": "raw_before_deployment_threshold",
                }
            )
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta_path


class AsyncImageWriter:
    def __init__(self, *, enabled: bool, num_workers: int = 2, max_queue_size: int = 512) -> None:
        self.enabled = enabled
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_queue_size)
        self._errors: list[str] = []
        self._error_lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        if not enabled:
            return
        for idx in range(num_workers):
            thread = threading.Thread(target=self._worker, name=f"episode-image-writer-{idx}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def write(self, path: Path, image_bgr: np.ndarray) -> None:
        if not self.enabled:
            return
        self._queue.put((path, image_bgr))

    def close(self) -> None:
        if not self.enabled:
            return
        self._queue.join()
        for _ in self._threads:
            self._queue.put(None)
        self._queue.join()
        for thread in self._threads:
            thread.join(timeout=3.0)

    def error_count(self) -> int:
        with self._error_lock:
            return len(self._errors)

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                path, image = item
                ok = cv2.imwrite(str(path), image)
                if not ok:
                    with self._error_lock:
                        self._errors.append(str(path))
            finally:
                self._queue.task_done()


def _csv_header(zed_eye: str) -> list[str]:
    return [
        "timestamp",
        f"image1_zed_{zed_eye}",
        "image2",
        "image3",
        "u_p1",
        "u_p2",
        "u_p3",
        "u_p4",
        "u_p5",
        "u_p6",
        "u_p7",
        "u_p8",
        "u_p9",
        "u_p10",
        "u_p11",
        "u_p12",
        "u_paw1",
        "u_paw2",
        "u_paw3",
        "u_paw4",
        "executed_action1",
        "executed_action2",
        "executed_action3",
        "executed_action4",
        "executed_action5",
        "executed_action6",
        "executed_action7",
        "x_pos1",
        "x_pos2",
        "x_pos3",
        "x_ang_radian1",
        "x_ang_radian2",
        "x_ang_radian3",
        "x_pos_vel1",
        "x_pos_vel2",
        "x_pos_vel3",
        "x_ang_radian_vel1",
        "x_ang_radian_vel2",
        "x_ang_radian_vel3",
    ]


def _pressure_state_csv_header(zed_eye: str) -> list[str]:
    return [
        "timestamp",
        f"image1_zed_{zed_eye}",
        "image2",
        "image3",
        "action_source",
        "intervention_active",
        *[f"u_p{i}" for i in range(1, 13)],
        *[f"u_paw{i}" for i in range(1, 5)],
        *[f"executed_action{i}" for i in range(1, 20)],
        *[f"vla_action{i}" for i in range(1, 20)],
        "x_pos1",
        "x_pos2",
        "x_pos3",
        "x_ang_radian1",
        "x_ang_radian2",
        "x_ang_radian3",
        "x_pos_vel1",
        "x_pos_vel2",
        "x_pos_vel3",
        "x_ang_radian_vel1",
        "x_ang_radian_vel2",
        "x_ang_radian_vel3",
    ]
