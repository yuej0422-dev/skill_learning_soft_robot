from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from soft_vla.schemas import ACTION_DIM, STATE_DIM, validate_action, validate_state


TASKS = [
    "Move the soft robot end effector to the colored target and close the gripper.",
    "Move the end effector to the target and close the gripper.",
    "Reach the colored target, then close the gripper.",
    "Approach the target with the soft robot and grasp it.",
]


@dataclass(frozen=True)
class SyntheticConfig:
    episodes: int = 12
    frames_per_episode: int = 40
    fps: int = 10
    image_height: int = 128
    image_width: int = 128
    seed: int = 42
    position_gain: float = 0.35
    rotation_gain: float = 0.30
    max_translation_step: float = 0.01
    max_rotation_step: float = 0.04


@dataclass
class EpisodeFrame:
    images: dict[str, np.ndarray]
    state: np.ndarray
    action: np.ndarray
    task: str
    timestamp: float
    episode_index: int
    frame_index: int
    target_position: np.ndarray
    target_rotation: np.ndarray


def _workspace_to_pixel_xy(pos: np.ndarray, w: int, h: int) -> tuple[int, int]:
    x = int(np.clip((pos[0] + 0.22) / 0.44 * (w - 1), 0, w - 1))
    y = int(np.clip((0.42 - pos[1]) / 0.64 * (h - 1), 0, h - 1))
    return x, y


def _workspace_to_pixel_xz(pos: np.ndarray, w: int, h: int) -> tuple[int, int]:
    x = int(np.clip((pos[0] + 0.22) / 0.44 * (w - 1), 0, w - 1))
    z = int(np.clip((0.42 - pos[2]) / 0.34 * (h - 1), 0, h - 1))
    return x, z


def _workspace_to_pixel_yz(pos: np.ndarray, w: int, h: int) -> tuple[int, int]:
    y = int(np.clip((pos[1] + 0.22) / 0.44 * (w - 1), 0, w - 1))
    z = int(np.clip((0.42 - pos[2]) / 0.34 * (h - 1), 0, h - 1))
    return y, z


def _draw_marker(draw: ImageDraw.ImageDraw, xy: tuple[int, int], radius: int, color: tuple[int, int, int]) -> None:
    x, y = xy
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def _render_view(
    view: str,
    tcp: np.ndarray,
    target: np.ndarray,
    trajectory: list[np.ndarray],
    target_color: tuple[int, int, int],
    height: int,
    width: int,
    gripper_closed: int,
) -> np.ndarray:
    base = (238, 240, 236) if view == "main" else ((230, 238, 246) if view == "wrist_left" else (242, 234, 230))
    image = Image.new("RGB", (width, height), base)
    draw = ImageDraw.Draw(image)
    margin = max(5, width // 18)
    draw.rectangle((margin, margin, width - margin, height - margin), outline=(70, 78, 88), width=1)

    if view == "main":
        project = lambda p: _workspace_to_pixel_xy(p, width, height)
    elif view == "wrist_left":
        project = lambda p: _workspace_to_pixel_yz(p - tcp + np.array([0.0, 0.22, 0.26]), width, height)
    else:
        project = lambda p: _workspace_to_pixel_xz(p, width, height)

    if len(trajectory) > 1:
        pts = [project(p) for p in trajectory[-16:]]
        draw.line(pts, fill=(95, 112, 132), width=1)

    target_xy = project(target)
    tcp_xy = project(tcp)
    distance = float(np.linalg.norm(target - tcp))
    target_radius = max(4, int(width * (0.08 if view == "wrist_left" else 0.055)))
    tcp_radius = max(3, int(width * 0.04))
    if view == "wrist_left":
        target_radius = max(4, int(width * np.clip(0.11 - distance * 0.15, 0.04, 0.12)))

    _draw_marker(draw, target_xy, target_radius, target_color)
    _draw_marker(draw, tcp_xy, tcp_radius, (28, 93, 155))
    jaw = (14, 14, 14) if gripper_closed else (255, 255, 255)
    draw.rectangle((tcp_xy[0] - tcp_radius, tcp_xy[1] + tcp_radius + 2, tcp_xy[0] + tcp_radius, tcp_xy[1] + tcp_radius + 5), fill=jaw)
    return np.asarray(image, dtype=np.uint8)


def generate_episode(config: SyntheticConfig, episode_index: int, rng: np.random.Generator) -> list[EpisodeFrame]:
    dt = 1.0 / config.fps
    position = rng.uniform([-0.18, -0.18, 0.12], [0.18, 0.18, 0.38]).astype(np.float32)
    rotation = rng.uniform(-0.25, 0.25, size=3).astype(np.float32)
    target_position = rng.uniform([-0.18, -0.18, 0.12], [0.18, 0.18, 0.38]).astype(np.float32)
    target_rotation = rng.uniform(-0.25, 0.25, size=3).astype(np.float32)
    colors = [(216, 57, 49), (58, 140, 76), (225, 176, 42), (128, 74, 170)]
    target_color = colors[episode_index % len(colors)]
    task = TASKS[episode_index % len(TASKS)]
    trajectory: list[np.ndarray] = []
    prev_position = position.copy()
    prev_rotation = rotation.copy()
    prev_gripper = 0
    frames: list[EpisodeFrame] = []

    for frame_index in range(config.frames_per_episode):
        position_error = target_position - position
        rotation_error = target_rotation - rotation
        delta_position = np.clip(
            config.position_gain * position_error,
            -config.max_translation_step,
            config.max_translation_step,
        )
        delta_rotation = np.clip(
            config.rotation_gain * rotation_error,
            -config.max_rotation_step,
            config.max_rotation_step,
        )
        delta_position += rng.normal(0.0, 0.0004, size=3).astype(np.float32)
        delta_rotation += rng.normal(0.0, 0.001, size=3).astype(np.float32)
        distance = float(np.linalg.norm(position_error))
        gripper_action = 1 if distance < 0.035 else 0

        linear_velocity = (position - prev_position) / dt if frame_index else np.zeros(3, dtype=np.float32)
        angular_velocity = (rotation - prev_rotation) / dt if frame_index else np.zeros(3, dtype=np.float32)
        gripper_state = prev_gripper
        state = np.concatenate(
            [position, rotation, linear_velocity, angular_velocity, np.array([gripper_state], dtype=np.float32)]
        ).astype(np.float32)
        action = np.concatenate([delta_position, delta_rotation, np.array([gripper_action], dtype=np.float32)]).astype(
            np.float32
        )
        validate_state(state)
        validate_action(action)

        trajectory.append(position.copy())
        images = {
            "observation.images.main": _render_view(
                "main", position, target_position, trajectory, target_color, config.image_height, config.image_width, gripper_state
            ),
            "observation.images.wrist_left": _render_view(
                "wrist_left",
                position,
                target_position,
                trajectory,
                target_color,
                config.image_height,
                config.image_width,
                gripper_state,
            ),
            "observation.images.wrist_right": _render_view(
                "wrist_right",
                position,
                target_position,
                trajectory,
                target_color,
                config.image_height,
                config.image_width,
                gripper_state,
            ),
        }
        frames.append(
            EpisodeFrame(
                images=images,
                state=state,
                action=action,
                task=task,
                timestamp=frame_index * dt,
                episode_index=episode_index,
                frame_index=frame_index,
                target_position=target_position.copy(),
                target_rotation=target_rotation.copy(),
            )
        )

        prev_position = position.copy()
        prev_rotation = rotation.copy()
        position = (position + delta_position).astype(np.float32)
        rotation = (rotation + delta_rotation).astype(np.float32)
        prev_gripper = gripper_action

    return frames


def generate_dataset(config: SyntheticConfig) -> list[list[EpisodeFrame]]:
    rng = np.random.default_rng(config.seed)
    return [generate_episode(config, ep, rng) for ep in range(config.episodes)]


def save_sample_mosaics(episodes: list[list[EpisodeFrame]], output_dir: str | Path, max_samples: int = 6) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for episode in episodes:
        for frame in episode[:: max(1, len(episode) // 2)]:
            imgs = [Image.fromarray(frame.images[k]) for k in sorted(frame.images)]
            w, h = imgs[0].size
            mosaic = Image.new("RGB", (w * len(imgs), h), (255, 255, 255))
            for i, img in enumerate(imgs):
                mosaic.paste(img, (i * w, 0))
            mosaic.save(out / f"episode_{frame.episode_index:06d}_frame_{frame.frame_index:06d}.png")
            count += 1
            if count >= max_samples:
                return

