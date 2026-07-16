from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CAMERA_KEYS = ("observation.images.cam_1", "observation.images.cam_2", "observation.images.cam_3")


@dataclass(frozen=True)
class LiveCameraConfig:
    zed_index: int | None = None
    zed_eye: str = "left"
    zed_width: int = 2560
    zed_height: int = 720
    zed_fps: int = 30
    realsense_width: int = 640
    realsense_height: int = 480
    realsense_fps: int = 30
    realsense_serial_cam2: str | None = None
    realsense_serial_cam3: str | None = None
    startup_timeout_s: float = 10.0
    zed_warmup_usable_frames: int = 10
    realsense_warmup_usable_frames: int = 10
    realsense_start_gap_s: float = 3.0
    min_gray_std: float = 2.0
    min_nonblack_fraction: float = 0.05
    min_realsense_mean: float = 40.0
    require_zed_device_name: bool = True


class OpenCVZEDLeftCamera:
    def __init__(self, config: LiveCameraConfig) -> None:
        self.config = config
        index = config.zed_index
        if index is None:
            index = find_zed_left_index(config)
        self.index = int(index)
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2
        self.cap = cv2.VideoCapture(self.index, backend)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open ZED RGB camera index {self.index}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.zed_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.zed_height)
        self.cap.set(cv2.CAP_PROP_FPS, config.zed_fps)
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, name="zed-left-camera", daemon=True)
        self._thread.start()
        self._wait_first_frame()

    def _wait_first_frame(self) -> None:
        deadline = time.time() + float(self.config.startup_timeout_s)
        last_quality = None
        usable_count = 0
        while time.time() < deadline:
            frame = self.read()
            if frame is not None:
                quality = image_quality(
                    bgr_to_rgb(frame),
                    min_gray_std=self.config.min_gray_std,
                    min_nonblack_fraction=self.config.min_nonblack_fraction,
                )
                if quality["usable"]:
                    usable_count += 1
                    if usable_count >= max(1, int(self.config.zed_warmup_usable_frames)):
                        return
                else:
                    usable_count = 0
                last_quality = quality
            time.sleep(0.05)
        self.release()
        if last_quality is None:
            raise RuntimeError(f"ZED camera index {self.index} did not produce a frame")
        raise RuntimeError(
            f"ZED camera index {self.index} did not produce a usable frame before timeout; "
            f"last_quality={last_quality}. Check that this is ZED left, not a green/black stream."
        )

    def _capture_loop(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            left = split_zed_eye(frame, self.config.zed_eye)
            with self._lock:
                self._latest = left.copy()

    def read(self) -> np.ndarray | None:
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy()

    def release(self) -> None:
        self._running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=3.0)
        if hasattr(self, "cap"):
            self.cap.release()


class RealSenseRGBCamera:
    def __init__(self, serial_number: str, config: LiveCameraConfig) -> None:
        import pyrealsense2 as rs

        self.rs = rs
        self.serial_number = serial_number
        self.config = config
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial_number)
        cfg.enable_stream(rs.stream.color, config.realsense_width, config.realsense_height, rs.format.bgr8, config.realsense_fps)
        cfg.enable_stream(rs.stream.depth, config.realsense_width, config.realsense_height, rs.format.z16, config.realsense_fps)
        self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)
        self._lock = threading.Lock()
        self._latest: tuple[np.ndarray, np.ndarray] | None = None
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, name=f"realsense-rgb-{serial_number}", daemon=True)
        self._thread.start()
        self._wait_first_frame()

    def _wait_first_frame(self) -> None:
        deadline = time.time() + float(self.config.startup_timeout_s)
        last_quality = None
        usable_count = 0
        while time.time() < deadline:
            frame = self.read()
            if frame is not None:
                color, _ = frame
                quality = realsense_image_quality(color, self.config)
                if quality["usable"]:
                    usable_count += 1
                    if usable_count >= max(1, int(self.config.realsense_warmup_usable_frames)):
                        return
                else:
                    usable_count = 0
                last_quality = quality
            time.sleep(0.05)
        self.release()
        if last_quality is None:
            raise RuntimeError(f"RealSense {self.serial_number} did not produce a frame")
        raise RuntimeError(
            f"RealSense {self.serial_number} did not produce a usable frame before timeout; "
            f"last_quality={last_quality}"
        )

    def _capture_loop(self) -> None:
        while self._running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError:
                continue
            aligned = self.align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color = np.asanyarray(color_frame.get_data()).copy()
            depth = np.asanyarray(depth_frame.get_data()).copy()
            with self._lock:
                self._latest = (color, depth)

    def read(self) -> tuple[np.ndarray, np.ndarray] | None:
        with self._lock:
            if self._latest is None:
                return None
            color, depth = self._latest
            return color.copy(), depth.copy()

    def release(self) -> None:
        self._running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=3.0)
        if hasattr(self, "pipeline"):
            self.pipeline.stop()


class LiveThreeCameraSource:
    def __init__(self, config: LiveCameraConfig | None = None) -> None:
        self.config = config or LiveCameraConfig()
        self.zed: OpenCVZEDLeftCamera | None = None
        self.cam2: RealSenseRGBCamera | None = None
        self.cam3: RealSenseRGBCamera | None = None
        self.realsense_serials: list[str] = []
        self.cam2_serial: str | None = None
        self.cam3_serial: str | None = None

    def open(self) -> None:
        try:
            self.zed = OpenCVZEDLeftCamera(self.config)
            serials = find_realsense_serials()
            self.realsense_serials = serials
            if self.config.realsense_serial_cam2 and self.config.realsense_serial_cam3:
                cam2_serial = self.config.realsense_serial_cam2
                cam3_serial = self.config.realsense_serial_cam3
            else:
                if len(serials) < 2:
                    raise RuntimeError(f"need 2 RealSense devices, got {serials}")
                # Prefer explicit serial binding. This fallback is for local smoke only:
                # pyrealsense2 enumeration order can differ from the Windows collection script.
                cam2_serial = serials[0]
                cam3_serial = serials[1]
            self.cam2 = RealSenseRGBCamera(cam2_serial, self.config)
            self.cam2_serial = cam2_serial
            time.sleep(float(self.config.realsense_start_gap_s))
            self.cam3 = RealSenseRGBCamera(cam3_serial, self.config)
            self.cam3_serial = cam3_serial
        except Exception:
            self.close()
            raise

    def read_rgb_uint8(self) -> dict[str, np.ndarray]:
        if self.zed is None or self.cam2 is None or self.cam3 is None:
            raise RuntimeError("live camera source is not open")
        zed_bgr = self.zed.read()
        cam2 = self.cam2.read()
        cam3 = self.cam3.read()
        if zed_bgr is None or cam2 is None or cam3 is None:
            raise RuntimeError("one or more cameras have no latest frame")
        cam2_bgr, _ = cam2
        cam3_bgr, _ = cam3
        images = {
            "observation.images.cam_1": bgr_to_rgb(ensure_shape(zed_bgr, (720, 1280))),
            "observation.images.cam_2": bgr_to_rgb(ensure_shape(cam2_bgr, (480, 640))),
            "observation.images.cam_3": bgr_to_rgb(ensure_shape(cam3_bgr, (480, 640))),
        }
        return images

    def read_policy_tensors(self) -> dict[str, Any]:
        import torch

        images = self.read_rgb_uint8()
        return {key: torch.from_numpy(rgb).permute(2, 0, 1).to(dtype=torch.float32) / 255.0 for key, rgb in images.items()}

    def smoke_report(self, output_dir: str | Path | None = None) -> dict[str, Any]:
        images = self.read_rgb_uint8()
        report = {
            "ok": True,
            "zed_index": None if self.zed is None else self.zed.index,
            "zed_eye": self.config.zed_eye,
            "zed_device_name": None if self.zed is None else video_device_name(self.zed.index),
            "zed_requested_width": self.config.zed_width,
            "zed_requested_height": self.config.zed_height,
            "zed_warmup_usable_frames": self.config.zed_warmup_usable_frames,
            "realsense_serials": self.realsense_serials,
            "realsense_cam2_serial": self.cam2_serial,
            "realsense_cam3_serial": self.cam3_serial,
            "realsense_warmup_usable_frames": self.config.realsense_warmup_usable_frames,
            "min_realsense_mean": self.config.min_realsense_mean,
            "cameras": {},
        }
        out = Path(output_dir) if output_dir is not None else None
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
        for key, rgb in images.items():
            item = camera_image_quality(key, rgb, self.config)
            report["cameras"][key] = item
            report["ok"] = bool(report["ok"] and item["usable"])
            if out is not None:
                cv2.imwrite(str(out / f"{key.rsplit('.', 1)[-1]}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if out is not None:
            (out / "camera_smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    def close(self) -> None:
        for camera in (self.cam3, self.cam2, self.zed):
            if camera is not None:
                camera.release()
        self.zed = None
        self.cam2 = None
        self.cam3 = None


def find_realsense_serials() -> list[str]:
    import pyrealsense2 as rs

    ctx = rs.context()
    return [dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()]


def find_zed_left_index(config: LiveCameraConfig, max_index: int = 16) -> int:
    device_candidates = zed_video_indices_from_sysfs()
    if device_candidates:
        return device_candidates[0]
    elif config.require_zed_device_name and os.name != "nt":
        raise RuntimeError("no /sys/class/video4linux device with name containing 'zed' was found")
    else:
        search_indices = list(range(max_index))
    candidates: list[tuple[float, int, dict[str, Any]]] = []
    rejected: list[dict[str, Any]] = []
    for index in search_indices:
        frame = read_opencv_frame(index, width=config.zed_width, height=config.zed_height, fps=config.zed_fps)
        if frame is None:
            rejected.append({"index": index, "reason": "no_frame"})
            continue
        left = split_zed_eye(frame, config.zed_eye)
        quality = image_quality(bgr_to_rgb(left), min_gray_std=config.min_gray_std, min_nonblack_fraction=config.min_nonblack_fraction)
        if quality["usable"]:
            score = float(quality["gray_std"]) + 0.001 * float(quality["unique_approx"])
            candidates.append((score, index, quality))
        else:
            rejected.append({"index": index, "reason": "unusable", "quality": quality})
    if not candidates:
        raise RuntimeError(f"no usable ZED RGB camera index found; rejected={rejected}")
    candidates.sort(reverse=True)
    return candidates[0][1]


def zed_video_indices_from_sysfs() -> list[int]:
    video_root = Path("/sys/class/video4linux")
    if not video_root.is_dir():
        return []
    indices: list[int] = []
    for entry in sorted(video_root.iterdir(), key=lambda p: p.name):
        if not entry.name.startswith("video"):
            continue
        try:
            index = int(entry.name.replace("video", ""))
        except ValueError:
            continue
        try:
            name = (entry / "name").read_text(encoding="utf-8", errors="ignore").strip().lower()
        except OSError:
            continue
        if "zed" in name:
            indices.append(index)
    return indices


def video_device_name(index: int) -> str | None:
    path = Path(f"/sys/class/video4linux/video{int(index)}/name")
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return None


def read_opencv_frame(index: int, *, width: int, height: int, fps: int) -> np.ndarray | None:
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    frame = None
    for _ in range(20):
        ok, maybe = cap.read()
        if ok and maybe is not None:
            frame = maybe
            break
        time.sleep(0.05)
    cap.release()
    return frame


def split_zed_eye(frame: np.ndarray, eye: str) -> np.ndarray:
    if eye not in {"left", "right"}:
        raise ValueError(f"zed eye must be left or right, got {eye}")
    h, w = frame.shape[:2]
    if w >= 2 * h:
        mid = w // 2
        return frame[:, :mid] if eye == "left" else frame[:, mid:]
    return frame


def ensure_shape(image: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    if image.shape[:2] == (h, w):
        return image
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image, dtype=np.uint8), cv2.COLOR_BGR2RGB)


def camera_image_quality(key: str, rgb: np.ndarray, config: LiveCameraConfig) -> dict[str, Any]:
    item = image_quality(rgb, min_gray_std=config.min_gray_std, min_nonblack_fraction=config.min_nonblack_fraction)
    if key in {"observation.images.cam_2", "observation.images.cam_3"}:
        apply_realsense_brightness_gate(item, config)
    return item


def realsense_image_quality(bgr: np.ndarray, config: LiveCameraConfig) -> dict[str, Any]:
    item = image_quality(bgr_to_rgb(bgr), min_gray_std=config.min_gray_std, min_nonblack_fraction=config.min_nonblack_fraction)
    apply_realsense_brightness_gate(item, config)
    return item


def apply_realsense_brightness_gate(item: dict[str, Any], config: LiveCameraConfig) -> None:
    min_mean = float(config.min_realsense_mean)
    if float(item["mean"]) < min_mean:
        item["usable"] = False
        item["reason"] = f"mean_below_min_realsense_mean:{min_mean:g}"


def image_quality(rgb: np.ndarray, *, min_gray_std: float, min_nonblack_fraction: float) -> dict[str, Any]:
    arr = np.asarray(rgb, dtype=np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mean_gradient = float(np.mean(np.sqrt(sobelx**2 + sobely**2)))
    sample = arr.reshape(-1, 3)[:: max(1, arr.shape[0] * arr.shape[1] // 10000)]
    gray_std = float(gray.std())
    nonblack_fraction = float(np.mean(gray > 5))
    unique_approx = int(len(np.unique(sample, axis=0)))
    usable = bool(gray_std >= float(min_gray_std) and nonblack_fraction >= float(min_nonblack_fraction) and unique_approx > 8)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "gray_std": gray_std,
        "laplacian_var": laplacian_var,
        "mean_gradient": mean_gradient,
        "nonblack_fraction": nonblack_fraction,
        "unique_approx": unique_approx,
        "usable": usable,
    }
