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
    - each CSV row stores the actually executed 7D deployment action.
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
            self._csv_writer.writerow(_csv_header(self.zed_eye))

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
        action7 = np.asarray(row.get("executed_action", np.zeros(7, dtype=np.float32)), dtype=np.float64).reshape(-1)
        u_p12 = np.asarray(row.get("u_p12", row.get("motion_norm12", np.zeros(12, dtype=np.float32))), dtype=np.float64).reshape(-1)
        u_paw4 = np.asarray(
            row.get("u_paw4", np.asarray(row.get("pressure16", np.zeros(16, dtype=np.float32)), dtype=np.float64).reshape(-1)[12:16]),
            dtype=np.float64,
        ).reshape(-1)
        if state12.shape[0] < 12 or action7.shape[0] != 7:
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
                    *u_p12.astype(float).tolist(),
                    *u_paw4.astype(float).tolist(),
                    *action7.astype(float).tolist(),
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
    """Keep the legacy human episode layout and add an aligned 25D/19D sidecar.

    ``data.csv`` and the three image directories are still written by
    :class:`HumanEpisodeSaver` without any schema changes.  The additional
    ``vla_training_data.csv`` stores observation ``t`` together with the
    realized transition from ``t`` to ``t + 1``:

    * state25 = state12_t + gripper_t + command_pressure12_t
    * action19 = (tcp6_{t+1} - tcp6_t) + commanded_gripper_t
      + (command_pressure12_{t+1} - command_pressure12_t) / pressure_delta_scale

    The final episode frame has no future observation and is intentionally not
    emitted as a training row.
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
        self.pressure_delta_scale = float(pressure_delta_scale)
        self.vla_training_csv_path = self.episode_dir / "vla_training_data.csv"
        self._vla_training_fh = (
            self.vla_training_csv_path.open("w", newline="", encoding="utf-8") if self.enabled else None
        )
        self._vla_training_writer = csv.writer(self._vla_training_fh) if self._vla_training_fh is not None else None
        if self._vla_training_writer is not None:
            self._vla_training_writer.writerow(_pressure_state_training_header(self.zed_eye))
        self._previous_training_frame: dict[str, Any] | None = None
        self._training_rows = 0

    def record_frame(self, row: dict[str, Any]) -> None:
        frame_idx = self.stats.frame_idx
        normalized = self._normalize_training_frame(row, frame_idx=frame_idx)
        super().record_frame(row)
        if normalized is None or self.stats.frame_idx != frame_idx + 1:
            return
        if self._previous_training_frame is not None:
            self._write_training_transition(self._previous_training_frame, normalized)
        self._previous_training_frame = normalized

    def close_episode(
        self,
        *,
        success: bool = False,
        failure: bool = False,
        termination_reason: str = "interrupted",
    ) -> Path:
        if self._vla_training_fh is not None:
            self._vla_training_fh.close()
            self._vla_training_fh = None
            self._vla_training_writer = None
        meta_path = super().close_episode(
            success=success,
            failure=failure,
            termination_reason=termination_reason,
        )
        if self.enabled:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta.update(
                {
                    "vla_training_data_csv": str(self.vla_training_csv_path),
                    "vla_training_rows": self._training_rows,
                    "vla_observation_dim": 25,
                    "vla_action_dim": 19,
                    "vla_transition_alignment": "observation_t_to_realized_transition_t_plus_1",
                    "pressure_delta_scale": self.pressure_delta_scale,
                }
            )
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta_path

    def _normalize_training_frame(self, row: dict[str, Any], *, frame_idx: int) -> dict[str, Any] | None:
        images = row.get("images")
        if not isinstance(images, dict):
            return None
        state12 = np.asarray(row.get("state12"), dtype=np.float64).reshape(-1)
        pressure12 = np.asarray(row.get("u_p12", row.get("motion_norm12")), dtype=np.float64).reshape(-1)
        if state12.shape != (12,) or pressure12.shape != (12,):
            return None
        if not np.all(np.isfinite(state12)) or not np.all(np.isfinite(pressure12)):
            return None
        gripper = float(row.get("gripper_open", row.get("executed_action_gripper", 1.0)))
        if not np.isfinite(gripper):
            return None
        return {
            "frame_idx": int(frame_idx),
            "timestamp": float(row.get("timestamp", frame_idx * 0.1)),
            "image_name": f"{frame_idx:06d}.jpg",
            "state12": state12,
            "pressure12": np.clip(pressure12, 0.0, 1.0),
            "gripper_open": 1.0 if gripper >= 0.5 else 0.0,
            "gripper_target": 1.0
            if float(row.get("executed_action_gripper", gripper)) >= 0.5
            else 0.0,
            "action_source": str(row.get("action_source", "unknown")),
            "intervention_active": bool(row.get("intervention_active", False)),
            "vla_action19": _fixed_vector(row.get("vla_action19"), 19),
            "commanded_delta_tcp6": _fixed_vector(row.get("executed_action_delta_tcp"), 6),
            "vla_feedforward_pressure12": _fixed_vector(row.get("vla_feedforward_pressure12"), 12),
            "closed_loop_delta_action12": _fixed_vector(row.get("closed_loop_delta_action12"), 12),
        }

    def _write_training_transition(self, current: dict[str, Any], nxt: dict[str, Any]) -> None:
        if self._vla_training_writer is None:
            return
        realized_delta_tcp6 = nxt["state12"][:6] - current["state12"][:6]
        raw_pressure_delta12 = nxt["pressure12"] - current["pressure12"]
        action_pressure_delta12 = raw_pressure_delta12 / self.pressure_delta_scale
        state25 = np.concatenate(
            [current["state12"], [current["gripper_open"]], current["pressure12"]]
        )
        action19 = np.concatenate(
            [realized_delta_tcp6, [current["gripper_target"]], action_pressure_delta12]
        )
        self._vla_training_writer.writerow(
            [
                current["timestamp"],
                current["image_name"],
                current["image_name"],
                current["image_name"],
                current["frame_idx"],
                nxt["frame_idx"],
                current["action_source"],
                int(current["intervention_active"]),
                *state25.astype(float).tolist(),
                *action19.astype(float).tolist(),
                *raw_pressure_delta12.astype(float).tolist(),
                *current["commanded_delta_tcp6"].astype(float).tolist(),
                *current["vla_action19"].astype(float).tolist(),
                *current["vla_feedforward_pressure12"].astype(float).tolist(),
                *current["closed_loop_delta_action12"].astype(float).tolist(),
            ]
        )
        self._training_rows += 1
        if self._training_rows % 10 == 0 and self._vla_training_fh is not None:
            self._vla_training_fh.flush()


def _fixed_vector(value: Any, size: int) -> np.ndarray:
    if value is None:
        return np.full(size, np.nan, dtype=np.float64)
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (size,):
        return np.full(size, np.nan, dtype=np.float64)
    return array


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


def _pressure_state_training_header(zed_eye: str) -> list[str]:
    state_names = [
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
        "gripper_open",
        *[f"u_p{i}" for i in range(1, 13)],
    ]
    action_names = [
        "delta_x_pos1",
        "delta_x_pos2",
        "delta_x_pos3",
        "delta_x_ang_radian1",
        "delta_x_ang_radian2",
        "delta_x_ang_radian3",
        "gripper_target",
        *[f"delta_u_p{i}" for i in range(1, 13)],
    ]
    return [
        "timestamp",
        f"image1_zed_{zed_eye}",
        "image2",
        "image3",
        "frame_index_t",
        "frame_index_t_plus_1",
        "action_source",
        "intervention_active",
        *[f"observation.state.{name}" for name in state_names],
        *[f"action.{name}" for name in action_names],
        *[f"raw_command_pressure_delta.{i}" for i in range(1, 13)],
        *[f"commanded_delta_tcp.{i}" for i in range(1, 7)],
        *[f"shadow_vla_action.{i}" for i in range(1, 20)],
        *[f"vla_feedforward_pressure.{i}" for i in range(1, 13)],
        *[f"closed_loop_delta_action.{i}" for i in range(1, 13)],
    ]
