import time
import threading
import numpy as np
import struct
import serial
import keyboard
import LuMoSDKClient
import pygame
import cv2
import os
import csv
import json
import argparse
import queue
import shutil
import pyrealsense2 as rs
from typing import Optional

# ──────────────────────────────────────────────
# RealSense D435 Helper（后台线程持续抓帧）
# ──────────────────────────────────────────────

class RealSenseCamera:
    """
    单台 D435 的封装，支持彩色 + 对齐深度流。
    使用后台线程持续抓帧，主线程调用 read() 时直接取最新帧，
    避免两台相机串行 wait_for_frames 时帧超时的问题。
    """

    def __init__(self, serial_number=None, width=640, height=480, fps=30):
        self.serial_number = serial_number

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        if serial_number:
            cfg.enable_device(serial_number)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.pipeline.start(cfg)

        # 最新帧缓存
        self._lock = threading.Lock()
        self._latest = None          # (color_image, depth_image, depth_colormap)
        self._running = True

        # 后台抓帧线程
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        # 等待后台线程拿到第一帧（最多 10 秒）
        deadline = time.time() + 10.0
        got_frame = False
        while time.time() < deadline:
            with self._lock:
                if self._latest is not None:
                    got_frame = True
                    break
            time.sleep(0.05)

        if not got_frame:
            raise RuntimeError("RealSense 相机 {} 启动超时，未收到帧".format(serial_number))

        print("相机 {} 已就绪".format(serial_number))

    def _capture_loop(self):
        """后台线程：持续从 pipeline 拉帧，保留最新一帧。"""
        align = rs.align(rs.stream.color)
        while self._running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=2000)
            except RuntimeError:
                continue

            aligned     = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )

            with self._lock:
                self._latest = (
                    color_image.copy(),
                    depth_image.copy(),
                    depth_colormap.copy()
                )

    def read(self):
        """
        返回 (color_bgr uint8, depth_uint16 mm, depth_colormap_bgr uint8)
        若后台线程还未拿到帧则返回 (None, None, None)
        """
        with self._lock:
            if self._latest is None:
                return None, None, None
            return self._latest

    def release(self):
        self._running = False
        self._thread.join(timeout=3.0)
        self.pipeline.stop()


def find_realsense_serials():
    """返回当前连接的所有 RealSense 设备序列号列表。"""
    ctx = rs.context()
    return [dev.get_info(rs.camera_info.serial_number)
            for dev in ctx.query_devices()]


def probe_camera_index(index, width=2560, height=720, fps=30):
    """打开一个 OpenCV 摄像头索引并返回首帧尺寸，失败则返回 None。"""
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
    for _ in range(10):
        ok, frame = cap.read()
        if ok and frame is not None:
            break
        time.sleep(0.05)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if frame is not None:
        frame_h, frame_w = frame.shape[:2]
        actual_w = frame_w or actual_w
        actual_h = frame_h or actual_h

    if actual_w <= 0 or actual_h <= 0:
        return None
    return actual_w, actual_h


def find_zed_camera_index(width=2560, height=720, fps=30, max_index=10):
    """自动查找 ZED 的 OpenCV 摄像头索引，Windows 下通过宽屏双目画面特征筛选。"""
    video_root = "/sys/class/video4linux"
    if os.path.isdir(video_root):
        candidates = []
        for entry in sorted(os.listdir(video_root)):
            if not entry.startswith("video"):
                continue
            try:
                index = int(entry.replace("video", ""))
            except ValueError:
                continue

            name_path = os.path.join(video_root, entry, "name")
            try:
                with open(name_path, "r", encoding="utf-8", errors="ignore") as f:
                    device_name = f.read().strip()
            except OSError:
                continue

            if "zed" in device_name.lower():
                candidates.append((index, device_name))

        if candidates:
            print("检测到 ZED 视频设备：{}".format(candidates))
            return candidates[0][0]

    probed = []
    for index in range(max_index):
        shape = probe_camera_index(index, width=width, height=height, fps=fps)
        if shape is None:
            continue

        actual_w, actual_h = shape
        aspect = actual_w / max(actual_h, 1)
        probed.append((index, actual_w, actual_h, round(aspect, 2)))
        if aspect >= 2.5:
            print("自动选择 ZED 摄像头 index:{} | {}x{} | aspect:{:.2f}".format(
                index, actual_w, actual_h, aspect
            ))
            return index

    if probed:
        print("已检测到 OpenCV 摄像头，但未发现 ZED 宽屏双目画面：{}".format(probed))
    return None


# ──────────────────────────────────────────────
# ZED 双目相机 Helper（OpenCV 读取 RGB，不依赖 pyzed.sl）
# ──────────────────────────────────────────────

class OpenCVZEDCamera:
    """
    ZED 双目相机 RGB 封装，使用 OpenCV / DirectShow 读取 USB 视频流。

    适用场景：只需要 ZED 的 RGB 图，不需要深度图，不想安装/调用 ZED SDK Python API。

    说明：
    - ZED 作为 USB 摄像头输出时，常见格式是左右目横向拼接的 side-by-side 图像；
      本类会自动按宽度一分为二，返回 left_bgr / right_bgr。
    - 如果你的 ZED 只输出单幅图，本类会把同一幅图同时作为 left/right 返回，保证主程序不报错。
    - Windows 下会自动扫描 OpenCV 摄像头索引，优先选择 ZED 常见的宽屏 side-by-side 输出；
      如果自动查找失败，请手动指定 ZED_CAMERA_INDEX 或命令行 --zed-index。
    """

    def __init__(self, camera_index=None, width=2560, height=720, fps=30):
        if camera_index is None:
            camera_index = find_zed_camera_index(width=width, height=height, fps=fps)
        if camera_index is None:
            raise RuntimeError(
                "未自动找到 ZED RGB 摄像头。Windows 下请确认 ZED 已作为 USB 摄像头连接，"
                "或手动尝试 --zed-index 1、2、3...，直到打开 ZED 而不是电脑内置摄像头。"
            )

        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps

        # Windows 下优先用 DirectShow；Linux 下优先用 V4L2，避免误开默认内置摄像头。
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2
        self.cap = cv2.VideoCapture(camera_index, backend)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(
                "ZED RGB 摄像头打开失败：camera_index={}。请确认 ZED 已连接，或修改 ZED_CAMERA_INDEX。".format(camera_index)
            )

        # 尽量设置为 ZED 常见 side-by-side 分辨率；实际是否生效由驱动决定。
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        self._lock = threading.Lock()
        self._latest = None          # (left_bgr, right_bgr)
        self._running = True

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        # 等待第一帧，最多 10 秒
        deadline = time.time() + 10.0
        got_frame = False
        while time.time() < deadline:
            with self._lock:
                if self._latest is not None:
                    got_frame = True
                    break
            time.sleep(0.05)

        if not got_frame:
            self.release()
            raise RuntimeError("ZED RGB 摄像头启动超时，未收到帧。可尝试更换 ZED_CAMERA_INDEX 或检查 USB3.0 连接。")

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print("ZED RGB 摄像头已就绪 | index:{} | {}x{} @ {:.1f}fps".format(
            camera_index, actual_w, actual_h, actual_fps
        ))

    @staticmethod
    def _split_stereo_frame(frame):
        """将 ZED side-by-side RGB 图像切成左右目。若不是横向双目，则复制为两份。"""
        if frame is None:
            return None, None

        h, w = frame.shape[:2]

        # ZED 常见输出是左右目横向拼接，宽度明显大于高度。
        # 按宽度一分为二，左半为 left，右半为 right。
        if w >= 2 * h:
            mid = w // 2
            left = frame[:, :mid]
            right = frame[:, mid:]
        else:
            # 如果驱动只给单目画面，则两路都保存同一帧，避免主流程中断。
            left = frame
            right = frame

        return left.copy(), right.copy()

    def _capture_loop(self):
        """后台线程：持续读取 ZED RGB 帧，保留最新一帧。"""
        while self._running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            left_bgr, right_bgr = self._split_stereo_frame(frame)
            if left_bgr is None or right_bgr is None:
                continue

            with self._lock:
                self._latest = (left_bgr, right_bgr)

    def read(self):
        """
        返回 (left_bgr, right_bgr)。
        若还未拿到帧则返回 (None, None)。
        """
        with self._lock:
            if self._latest is None:
                return None, None
            return self._latest

    def release(self):
        self._running = False
        if hasattr(self, '_thread'):
            self._thread.join(timeout=3.0)
        if hasattr(self, 'cap'):
            self.cap.release()


# ──────────────────────────────────────────────
# 控制辅助函数（与原版完全一致）
# ──────────────────────────────────────────────

def encode_xbox(u_p, input_mode, velity):
    # 获取摇杆的偏移量
    left_stick_x = joystick.get_axis(LEFT_STICK_X)
    left_stick_y = joystick.get_axis(LEFT_STICK_Y)
    right_stick_x = joystick.get_axis(RIGHT_STICK_X)
    right_stick_y = joystick.get_axis(RIGHT_STICK_Y)

    # 获取扳机和肩键
    left_trigger = joystick.get_axis(LEFT_TRIGGER)
    left_shoulder = joystick.get_button(LEFT_SHOULDER)
    right_trigger = joystick.get_axis(RIGHT_TRIGGER)
    right_shoulder = joystick.get_button(RIGHT_SHOULDER)
    if input_mode == 0:
        # ==========1seg============
        u_p[0] += left_stick_x * velity if abs(left_stick_x) > 0.1 else 0
        u_p[1] -= left_stick_x * velity if abs(left_stick_x) > 0.1 else 0
        u_p[2] -= left_stick_x * velity if abs(left_stick_x) > 0.1 else 0
        u_p[3] += left_stick_x * velity if abs(left_stick_x) > 0.1 else 0
        u_p[0] -= left_stick_y * velity if abs(left_stick_y) > 0.1 else 0
        u_p[1] -= left_stick_y * velity if abs(left_stick_y) > 0.1 else 0
        u_p[2] += left_stick_y * velity if abs(left_stick_y) > 0.1 else 0
        u_p[3] += left_stick_y * velity if abs(left_stick_y) > 0.1 else 0
        # 通过左扳机和左肩键调整
        if left_trigger > -0.9 and right_trigger < -0.9:  # 左扳机按下时增加
            u_p[0] += velity
            u_p[1] += velity
            u_p[2] += velity
            u_p[3] += velity
        if left_shoulder and not right_shoulder:  # 左肩键按下时减少
            u_p[0] -= velity
            u_p[1] -= velity
            u_p[2] -= velity
            u_p[3] -= velity
        # ==========2seg============
        u_p[4] += right_stick_x * velity if abs(right_stick_x) > 0.1 else 0
        u_p[5] -= right_stick_x * velity if abs(right_stick_x) > 0.1 else 0
        u_p[6] -= right_stick_x * velity if abs(right_stick_x) > 0.1 else 0
        u_p[7] += right_stick_x * velity if abs(right_stick_x) > 0.1 else 0
        u_p[4] -= right_stick_y * velity if abs(right_stick_y) > 0.1 else 0
        u_p[5] -= right_stick_y * velity if abs(right_stick_y) > 0.1 else 0
        u_p[6] += right_stick_y * velity if abs(right_stick_y) > 0.1 else 0
        u_p[7] += right_stick_y * velity if abs(right_stick_y) > 0.1 else 0
        # 通过左扳机和左肩键调整
        if right_trigger > -0.9 and left_trigger < -0.9:  # 左扳机按下时增加
            u_p[4] += velity
            u_p[5] += velity
            u_p[6] += velity
            u_p[7] += velity
        if right_shoulder and not left_shoulder:  # 左肩键按下时减少
            u_p[4] -= velity
            u_p[5] -= velity
            u_p[6] -= velity
            u_p[7] -= velity
        # ==========3seg============
        hat = joystick.get_hat(0)  # 参数0表示第一个方向键
        x, y = hat
        u_p[8] += x * velity if abs(x) > 0.1 else 0
        u_p[9] -= x * velity if abs(x) > 0.1 else 0
        u_p[10] -= x * velity if abs(x) > 0.1 else 0
        u_p[11] += x * velity if abs(x) > 0.1 else 0
        u_p[8] += y * velity if abs(y) > 0.1 else 0
        u_p[9] += y * velity if abs(y) > 0.1 else 0
        u_p[10] -= y * velity if abs(y) > 0.1 else 0
        u_p[11] -= y * velity if abs(y) > 0.1 else 0
        # 通过左扳机和左肩键调整
        if left_trigger > -0.9 and right_trigger > -0.9:  # 左扳机按下时增加
            u_p[8] += velity
            u_p[9] += velity
            u_p[10] += velity
            u_p[11] += velity
        if left_shoulder and right_shoulder:  # 左肩键按下时减少
            u_p[8] -= velity
            u_p[9] -= velity
            u_p[10] -= velity
            u_p[11] -= velity
    return u_p


def encode_keyboard(u_p, input_mode, interval_step):
    if input_mode == 1:
        if keyboard.is_pressed('shift'):
            if keyboard.is_pressed('a'):
                u_p[0] += interval_step
            if keyboard.is_pressed('s'):
                u_p[1] += interval_step
            if keyboard.is_pressed('w'):
                u_p[2] += interval_step
            if keyboard.is_pressed('q'):
                u_p[3] += interval_step
            if keyboard.is_pressed('d'):
                u_p[4] += interval_step
            if keyboard.is_pressed('f'):
                u_p[5] += interval_step
            if keyboard.is_pressed('r'):
                u_p[6] += interval_step
            if keyboard.is_pressed('e'):
                u_p[7] += interval_step
            if keyboard.is_pressed('g'):
                u_p[8] += interval_step
            if keyboard.is_pressed('h'):
                u_p[9] += interval_step
            if keyboard.is_pressed('y'):
                u_p[10] += interval_step
            if keyboard.is_pressed('t'):
                u_p[11] += interval_step
        else:
            if keyboard.is_pressed('a'):
                u_p[0] -= interval_step
            if keyboard.is_pressed('s'):
                u_p[1] -= interval_step
            if keyboard.is_pressed('w'):
                u_p[2] -= interval_step
            if keyboard.is_pressed('q'):
                u_p[3] -= interval_step
            if keyboard.is_pressed('d'):
                u_p[4] -= interval_step
            if keyboard.is_pressed('f'):
                u_p[5] -= interval_step
            if keyboard.is_pressed('r'):
                u_p[6] -= interval_step
            if keyboard.is_pressed('e'):
                u_p[7] -= interval_step
            if keyboard.is_pressed('g'):
                u_p[8] -= interval_step
            if keyboard.is_pressed('h'):
                u_p[9] -= interval_step
            if keyboard.is_pressed('y'):
                u_p[10] -= interval_step
            if keyboard.is_pressed('t'):
                u_p[11] -= interval_step
    return u_p


def paw_encode(u_paw):
    if joystick.get_button(Y_BUTTON):
        u_paw[0] = 3
        u_paw[1] = 0
    if joystick.get_button(A_BUTTON):
        u_paw[0] = 0
        u_paw[1] = 3
    return u_paw


# ──────────────────────────────────────────────
# 基础配置 / 工具函数
# ──────────────────────────────────────────────

SAMPLING_INTERVAL = 0.1
ROOT_DIR = "robot_records_7_03_1"
NUM_EPISODES = 50
RESET_SECONDS = 7.0
ZED_CAMERA_INDEX = None
ZED_EYE = "left"
SERIAL_PORT = "COM3"
IMAGE_WRITER_WORKERS = 2
IMAGE_WRITER_QUEUE_SIZE = 512


def parse_args():
    parser = argparse.ArgumentParser(description="Collect soft robot VLA episodes from 3 cameras.")
    parser.add_argument("--root-dir", default=ROOT_DIR, help="episode 根目录")
    parser.add_argument("--num-episodes", type=int, default=NUM_EPISODES, help="本次运行采集 episode 数")
    parser.add_argument("--sampling-interval", type=float, default=SAMPLING_INTERVAL, help="采样间隔，单位秒")
    parser.add_argument("--reset-seconds", type=float, default=RESET_SECONDS, help="成功后全零动作复位时长")
    parser.add_argument("--zed-index", type=int, default=ZED_CAMERA_INDEX, help="ZED 的 OpenCV 摄像头索引")
    parser.add_argument("--zed-eye", choices=["left", "right"], default=ZED_EYE, help="保存 ZED 左目或右目")
    parser.add_argument("--serial-port", default=SERIAL_PORT, help="机器人串口")
    parser.add_argument("--image-writer-workers", type=int, default=IMAGE_WRITER_WORKERS, help="后台写图片线程数")
    parser.add_argument("--image-writer-queue-size", type=int, default=IMAGE_WRITER_QUEUE_SIZE, help="后台写图片队列长度")
    return parser.parse_args()


def pack_action(u_p, u_paw):
    return struct.pack(
        "dddddddddddddddd",
        3 * u_p[0], 3 * u_p[1], 3 * u_p[2], 3 * u_p[3],
        3 * u_p[4], 3 * u_p[5], 3 * u_p[6], 3 * u_p[7],
        3 * u_p[8], 3 * u_p[9], 3 * u_p[10], 3 * u_p[11],
        u_paw[0], u_paw[1], u_paw[2], u_paw[3],
    )


def zero_action_buffer():
    return struct.pack("dddddddddddddddd", *([0.0] * 16))


def send_zero_action(serial_port, duration=0.0, interval=0.05):
    buffer = zero_action_buffer()
    if duration <= 0:
        serial_port.write(buffer)
        return

    end_time = time.time() + duration
    while time.time() < end_time:
        pygame.event.pump()
        serial_port.write(buffer)
        time.sleep(interval)


def find_next_episode_index(root_dir):
    indexes = []
    if os.path.isdir(root_dir):
        for name in os.listdir(root_dir):
            if not name.startswith("episode_"):
                continue
            try:
                indexes.append(int(name.split("_", 1)[1]))
            except ValueError:
                continue
    return max(indexes) + 1 if indexes else 1


def open_episode(root_dir, episode_index, zed_eye):
    save_dir = os.path.join(root_dir, "episode_{:03d}".format(episode_index))
    paths = {
        "save_dir": save_dir,
        "img_cam1": os.path.join(save_dir, "images_cam1_zed_{}".format(zed_eye)),
        "img_cam2": os.path.join(save_dir, "images_cam2"),
        "img_cam3": os.path.join(save_dir, "images_cam3"),
        "depth_cam2": os.path.join(save_dir, "depth_cam2"),
        "depth_cam3": os.path.join(save_dir, "depth_cam3"),
        "csv": os.path.join(save_dir, "data.csv"),
    }

    os.makedirs(save_dir, exist_ok=False)
    for key in ["img_cam1", "img_cam2", "img_cam3", "depth_cam2", "depth_cam3"]:
        os.makedirs(paths[key], exist_ok=False)

    csv_file = open(paths["csv"], "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestamp",
        "image1_zed_{}".format(zed_eye),
        "image2", "depth2",
        "image3", "depth3",
        "u_p1", "u_p2", "u_p3", "u_p4",
        "u_p5", "u_p6", "u_p7", "u_p8",
        "u_p9", "u_p10", "u_p11", "u_p12",
        "u_paw1", "u_paw2", "u_paw3", "u_paw4",
        "x_pos1", "x_pos2", "x_pos3",
        "x_ang_radian1", "x_ang_radian2", "x_ang_radian3",
        "x_pos_vel1", "x_pos_vel2", "x_pos_vel3",
        "x_ang_radian_vel1", "x_ang_radian_vel2", "x_ang_radian_vel3",
    ])
    return paths, csv_file, csv_writer


def write_episode_meta(save_dir, success, reason, frame_count, duration,
                       zed_eye=None, sampling_interval=None):
    if save_dir is None:
        return
    meta = {
        "success": bool(success),
        "reason": reason,
        "frame_count": int(frame_count),
        "duration_sec": round(float(duration), 4),
    }
    if zed_eye is not None:
        meta["zed_eye"] = zed_eye
    if sampling_interval is not None:
        meta["sampling_interval_sec"] = float(sampling_interval)
    with open(os.path.join(save_dir, "episode_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


class AsyncImageWriter:
    """后台写图片，避免 cv2.imwrite 阻塞采样节拍。"""

    def __init__(self, num_workers=2, max_queue_size=512):
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._errors = []
        self._error_lock = threading.Lock()
        self._threads = []
        for idx in range(num_workers):
            thread = threading.Thread(target=self._worker, name="image-writer-{}".format(idx), daemon=True)
            thread.start()
            self._threads.append(thread)

    def _worker(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                path, image = item
                ok = cv2.imwrite(path, image)
                if not ok:
                    with self._error_lock:
                        self._errors.append(path)
            finally:
                self._queue.task_done()

    def write(self, path, image):
        self._queue.put((path, image))

    def pending(self):
        return self._queue.qsize()

    def wait(self):
        self._queue.join()

    def close(self):
        self.wait()
        for _ in self._threads:
            self._queue.put(None)
        self._queue.join()
        for thread in self._threads:
            thread.join(timeout=3.0)

    def error_count(self):
        with self._error_lock:
            return len(self._errors)


# ──────────────────────────────────────────────
# pygame / 手柄初始化
# ──────────────────────────────────────────────

args = parse_args()
if args.num_episodes <= 0:
    raise ValueError("--num-episodes 必须大于 0")
if args.sampling_interval <= 0:
    raise ValueError("--sampling-interval 必须大于 0")
if args.image_writer_workers <= 0:
    raise ValueError("--image-writer-workers 必须大于 0")
if args.image_writer_queue_size <= 0:
    raise ValueError("--image-writer-queue-size 必须大于 0")
os.makedirs(args.root_dir, exist_ok=True)

pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    print("未检测到手柄。")
    exit()
joystick = pygame.joystick.Joystick(0)
joystick.init()

LEFT_STICK_X = 0
LEFT_STICK_Y = 1
RIGHT_STICK_X = 2
RIGHT_STICK_Y = 3
LEFT_TRIGGER = 4
RIGHT_TRIGGER = 5

# 按钮
A_BUTTON = 0
B_BUTTON = 1
X_BUTTON = 2
Y_BUTTON = 3
LEFT_SHOULDER = 4
RIGHT_SHOULDER = 5
MENU_BUTTON = 6
VIEW_BUTTON = 7

# 方向键
D_PAD_UP = 8
D_PAD_DOWN = 9
D_PAD_LEFT = 10
D_PAD_RIGHT = 11

# 摇杆按钮
LEFT_STICK_BUTTON = 12
RIGHT_STICK_BUTTON = 13
# ──────────────────────────────────────────────
# 串口初始化
# ──────────────────────────────────────────────

ser = serial.Serial(args.serial_port, 115200)
if ser.isOpen():
    print("串口打开成功：", ser.name)
else:
    print("串口打开失败。")

# ──────────────────────────────────────────────
# 摄像头初始化：cam1 为 ZED
# ──────────────────────────────────────────────

cam1 = None
cam2 = None
cam3 = None
image_writer = None
serials = find_realsense_serials()
print("检测到 {} 台 RealSense 相机：{}".format(len(serials), serials))
if len(serials) < 2:
    raise RuntimeError("需要两台 RealSense：cam2 使用原 cam2，cam3 使用原 cam1。")

cam1 = OpenCVZEDCamera(camera_index=args.zed_index, width=2560, height=720, fps=30)
time.sleep(3)
cam2 = RealSenseCamera(serial_number=serials[1])
time.sleep(3)   # 两台 RealSense 之间留出 USB 枚举间隔
cam3 = RealSenseCamera(serial_number=serials[0])
image_writer = AsyncImageWriter(
    num_workers=args.image_writer_workers,
    max_queue_size=args.image_writer_queue_size,
)

velity = 0.025
ip = "192.168.140.1"

LuMoSDKClient.Init()
LuMoSDKClient.Connnect(ip)

u_input = 0.0 * np.array(
    [0.       ,  0.     ,    1.      ,   0.74375354 ,0.25776123, 0.55489868,
 0.55223877, 0.25510132 ,0.06  ,     0.34    ,   0.32   ,    0.04      ,
     2.9, 2.9, 0., 0.], dtype=np.float64)

u_p = u_input[:12]
u_paw = u_input[12:]
input_mode = 0
time.sleep(5)

print("===== 操作说明 =====")
print("本次运行采集 {} 条 episode，根目录：{}".format(args.num_episodes, args.root_dir))
print("采样间隔：{:.3f}s".format(args.sampling_interval))
print("后台写图：{} workers，队列长度 {}".format(
    args.image_writer_workers, args.image_writer_queue_size
))
print("X：当前 episode 成功，结束本条并全零动作复位 {:.1f}s 后进入下一条".format(args.reset_seconds))
print("B：丢弃当前 episode，复位后重新开始一条")
print("ESC：中止采集并发送全零动作")
print("====================")
x_pos_list        = np.zeros((3, 1))
x_ang_radian_list = np.zeros((3, 1))
latest_sensor_data = np.zeros(12, dtype=np.float64)

episode_index = find_next_episode_index(args.root_dir)
episodes_done = 0
episode_paths = None
csv_file = None
csv_writer = None
frame_idx = 0
sample_slot = 0
next_sample_time = time.time()
record_start_time = 0.0
prev_x_pressed = False
prev_b_pressed = False

episode_paths, csv_file, csv_writer = open_episode(args.root_dir, episode_index, args.zed_eye)
frame_idx = 0
sample_slot = 0
record_start_time = time.time()
next_sample_time = record_start_time
print("\n=== 开始录制：episode_{:03d} | ZED {} ===".format(episode_index, args.zed_eye))

try:
    while episodes_done < args.num_episodes:
        pygame.event.pump()
        x_down = joystick.get_button(X_BUTTON)
        b_down = joystick.get_button(B_BUTTON)
        x_pressed = x_down and not prev_x_pressed
        b_pressed = b_down and not prev_b_pressed
        prev_x_pressed = x_down
        prev_b_pressed = b_down

        if keyboard.is_pressed('esc'):
            duration = time.time() - record_start_time
            if csv_file:
                csv_file.close()
                csv_file = None
            write_episode_meta(
                episode_paths["save_dir"], False, "interrupted", frame_idx,
                duration, args.zed_eye, args.sampling_interval
            )
            print("\n退出采集 | 当前 episode 已标记为 interrupted | 帧数：{}".format(frame_idx))
            u_p[:] = 0.0
            u_paw[:] = 0.0
            send_zero_action(ser, 0.0)
            break

        if b_pressed:
            duration = time.time() - record_start_time
            discarded_dir = episode_paths["save_dir"]
            if csv_file:
                csv_file.close()
                csv_file = None
            write_episode_meta(
                discarded_dir, False, "discard_button_b", frame_idx,
                duration, args.zed_eye, args.sampling_interval
            )
            print("\n=== episode_{:03d} 丢弃 | 帧数：{} | 时长：{:.2f}s ===".format(
                episode_index, frame_idx, duration
            ))

            if image_writer is not None:
                pending = image_writer.pending()
                if pending:
                    print("等待后台写图完成后删除当前 episode，剩余任务：{}".format(pending))
                image_writer.wait()

            if os.path.isdir(discarded_dir):
                shutil.rmtree(discarded_dir)
                print("已删除：{}".format(discarded_dir))

            u_p[:] = 0.0
            u_paw[:] = 0.0
            print("全零动作复位 {:.1f}s...".format(args.reset_seconds))
            send_zero_action(ser, args.reset_seconds)

            episode_paths, csv_file, csv_writer = open_episode(args.root_dir, episode_index, args.zed_eye)
            frame_idx = 0
            sample_slot = 0
            record_start_time = time.time()
            next_sample_time = record_start_time
            print("\n=== 重新开始录制：episode_{:03d} | ZED {} ===".format(episode_index, args.zed_eye))
            continue

        if x_pressed:
            duration = time.time() - record_start_time
            if csv_file:
                csv_file.close()
                csv_file = None
            write_episode_meta(
                episode_paths["save_dir"], True, "success_button_x", frame_idx,
                duration, args.zed_eye, args.sampling_interval
            )
            episodes_done += 1
            print("\n=== episode_{:03d} 成功结束 | 帧数：{} | 时长：{:.2f}s ===".format(
                episode_index, frame_idx, duration
            ))

            u_p[:] = 0.0
            u_paw[:] = 0.0
            print("全零动作复位 {:.1f}s...".format(args.reset_seconds))
            send_zero_action(ser, args.reset_seconds)

            if episodes_done >= args.num_episodes:
                print("已完成 {} 条 episode。".format(episodes_done))
                break

            episode_index += 1
            episode_paths, csv_file, csv_writer = open_episode(args.root_dir, episode_index, args.zed_eye)
            frame_idx = 0
            sample_slot = 0
            record_start_time = time.time()
            next_sample_time = record_start_time
            print("\n=== 开始录制：episode_{:03d} | ZED {} ===".format(episode_index, args.zed_eye))
            continue

        if joystick.get_button(A_BUTTON):
            input_mode = 0
        u_p = encode_xbox(u_p, input_mode, velity)
        u_p = encode_keyboard(u_p, input_mode, velity)
        u_p = np.clip(u_p, 0, 0.9)
        u_paw = paw_encode(u_paw)
        u_paw = np.clip(u_paw, 0, 3)
        # 打包数据并写入串口
       # print(u_p)

        buffer = pack_action(u_p, u_paw)
        write_len = ser.write(buffer)

        # ── 运动捕捉数据 ──
        frame = LuMoSDKClient.ReceiveData(0)
        sensor_data = latest_sensor_data
        if frame is not None:
            for rigid in frame.rigidBodys:
                if rigid.Id == 1:
                    sensor_data = np.array([
                    0.001 * rigid.X, 0.001 * rigid.Y, 0.001 * rigid.Z,
                    rigid.speeds.XfSpeed, rigid.speeds.YfSpeed, rigid.speeds.ZfSpeed,
                    np.pi * rigid.eulerAngle.X / 180.0,
                    np.pi * rigid.eulerAngle.Y / 180.0,
                    np.pi * rigid.eulerAngle.Z / 180.0,
                    rigid.palstance.fXPalstance,
                    rigid.palstance.fYPalstance,
                    rigid.palstance.fZPalstance,
                    ], dtype=np.float64)
                    latest_sensor_data = sensor_data
                    break
        x_pos             = sensor_data[0:3].reshape(3, 1)
        x_ang_radian      = sensor_data[6:9].reshape(3, 1)
        x_pos_list        = np.concatenate((x_pos_list, x_pos), axis=1)
        x_ang_radian_list = np.concatenate((x_ang_radian_list, x_ang_radian), axis=1)
        x_ang_radian_vel  = (x_ang_radian_list[:, -1] - x_ang_radian_list[:, -2]) / 0.02
        x_pos_vel         = sensor_data[3:6].reshape(3, 1)
        x = np.concatenate((
            x_pos.reshape(1, 3), x_ang_radian.reshape(1, 3),
            x_pos_vel.reshape(1, 3), x_ang_radian_vel.reshape(1, 3)
        ), axis=1).flatten()

        # ── 录制采样 ──
        now = time.time()
        if now >= next_sample_time:
            scheduled_elapsed = round(sample_slot * args.sampling_interval, 4)
            sample_lag = now - next_sample_time
            sample_slot += 1
            next_sample_time = record_start_time + sample_slot * args.sampling_interval

            zed_left, zed_right = cam1.read()
            color2, depth2_raw, depth2_vis = cam2.read()
            color3, depth3_raw, depth3_vis = cam3.read()
            zed_image = zed_left if args.zed_eye == "left" else zed_right

            if (zed_image is not None and color2 is not None and color3 is not None and
                    depth2_raw is not None and depth3_raw is not None):
                img_name   = "{:06d}.jpg".format(frame_idx)
                depth_name = "{:06d}.png".format(frame_idx)   # 16-bit PNG，单位 mm

                image_writer.write(os.path.join(episode_paths["img_cam1"], img_name), zed_image)
                image_writer.write(os.path.join(episode_paths["img_cam2"], img_name), color2)
                image_writer.write(os.path.join(episode_paths["img_cam3"], img_name), color3)
                image_writer.write(os.path.join(episode_paths["depth_cam2"], depth_name), depth2_raw)
                image_writer.write(os.path.join(episode_paths["depth_cam3"], depth_name), depth3_raw)

                control      = np.concatenate([u_p, u_paw])
                csv_writer.writerow([
                    scheduled_elapsed,
                    img_name,
                    img_name, depth_name,
                    img_name, depth_name,
                    *control, *x
                ])
                if frame_idx % 10 == 0:
                    csv_file.flush()
                print("帧:{:05d} | t:{:.2f}s | lag:{:.1f}ms | write_q:{} | cam1_ZED_{}:{} | cam2:{} depth2:{} | cam3:{} depth3:{}".format(
                    frame_idx, scheduled_elapsed, 1000.0 * sample_lag,
                    image_writer.pending(), args.zed_eye, img_name, img_name, depth_name, img_name, depth_name
                ))
                frame_idx += 1
            else:
                if zed_image is None:
                    print("警告：相机1/ZED {} 未读取到RGB帧，跳过".format(args.zed_eye))
                if color2 is None:
                    print("警告：相机2/RealSense未读取到帧，跳过")
                if color3 is None:
                    print("警告：相机3/RealSense未读取到帧，跳过")
                if depth2_raw is None:
                    print("警告：相机2/RealSense未读取到深度帧，跳过")
                if depth3_raw is None:
                    print("警告：相机3/RealSense未读取到深度帧，跳过")

        sleep_time = max(0.0, next_sample_time - time.time())
        if sleep_time > 0:
            time.sleep(sleep_time)

finally:
    if csv_file:
        duration = time.time() - record_start_time
        csv_file.close()
        write_episode_meta(
            episode_paths["save_dir"], False, "closed_in_finally", frame_idx,
            duration, args.zed_eye, args.sampling_interval
        )
    if image_writer is not None:
        pending = image_writer.pending()
        if pending:
            print("等待后台写图完成，剩余任务：{}".format(pending))
        image_writer.close()
        if image_writer.error_count():
            print("警告：有 {} 张图片写入失败，请检查磁盘或路径权限。".format(image_writer.error_count()))
    if cam1 is not None:
        cam1.release()
    if cam2 is not None:
        cam2.release()
    if cam3 is not None:
        cam3.release()
    cv2.destroyAllWindows()
    send_zero_action(ser, 0.0)
    if ser.isOpen():
        ser.close()
    print("资源已释放")
