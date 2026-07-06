from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


STATE_NAMES = [
    "tcp_x",
    "tcp_y",
    "tcp_z",
    "tcp_rx",
    "tcp_ry",
    "tcp_rz",
    "tcp_vx",
    "tcp_vy",
    "tcp_vz",
    "tcp_wx",
    "tcp_wy",
    "tcp_wz",
    "gripper_state",
]

STATE_UNITS = [
    "m",
    "m",
    "m",
    "rad",
    "rad",
    "rad",
    "m/s",
    "m/s",
    "m/s",
    "rad/s",
    "rad/s",
    "rad/s",
    "binary",
]

ACTION_NAMES = [
    "delta_tcp_x",
    "delta_tcp_y",
    "delta_tcp_z",
    "delta_tcp_rx",
    "delta_tcp_ry",
    "delta_tcp_rz",
    "gripper_action",
]

ACTION_UNITS = ["m", "m", "m", "rad", "rad", "rad", "binary"]

STATE_DIM = 13
ACTION_DIM = 7
GRIPPER_STATE_INDEX = 12
GRIPPER_ACTION_INDEX = 6


@dataclass(frozen=True)
class StateSchema:
    names: tuple[str, ...] = tuple(STATE_NAMES)
    units: tuple[str, ...] = tuple(STATE_UNITS)
    tcp_pose_representation: str = "rotation_vector"
    gripper_mode: str = "binary_open_close"


@dataclass(frozen=True)
class ActionSchema:
    names: tuple[str, ...] = tuple(ACTION_NAMES)
    units: tuple[str, ...] = tuple(ACTION_UNITS)
    is_delta_tcp: bool = True
    rotation_representation: str = "rotation_vector"
    gripper_mode: str = "binary_absolute_target_position"


def _as_array(value: np.ndarray | Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 0:
        raise ValueError(f"{name} must have a trailing dimension.")
    return arr


def _require_binary(values: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or Inf.")
    if not np.all((values == 0) | (values == 1)):
        unique = np.unique(values)
        raise ValueError(f"{name} must be binary 0/1. Found values: {unique[:10]}.")


def validate_state(state: np.ndarray | Iterable[float], *, require_binary_gripper: bool = True) -> np.ndarray:
    arr = _as_array(state, "observation.state")
    if arr.shape[-1] != STATE_DIM:
        raise ValueError(f"observation.state must end with dim {STATE_DIM}, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("observation.state contains NaN or Inf.")
    if require_binary_gripper:
        _require_binary(arr[..., GRIPPER_STATE_INDEX], "state[12] gripper_state")
    return arr


def validate_action(action: np.ndarray | Iterable[float], *, require_binary_gripper: bool = True) -> np.ndarray:
    arr = _as_array(action, "action")
    if arr.shape[-1] != ACTION_DIM:
        raise ValueError(f"action must end with dim {ACTION_DIM}, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("action contains NaN or Inf.")
    if require_binary_gripper:
        _require_binary(arr[..., GRIPPER_ACTION_INDEX], "action[6] gripper_action")
    return arr


def lerobot_features(image_height: int, image_width: int, *, use_videos: bool = False) -> dict:
    image_dtype = "video" if use_videos else "image"
    image_shape = (image_height, image_width, 3)
    image_names = ["height", "width", "channels"]
    return {
        "observation.images.main": {"dtype": image_dtype, "shape": image_shape, "names": image_names},
        "observation.images.wrist_left": {"dtype": image_dtype, "shape": image_shape, "names": image_names},
        "observation.images.wrist_right": {"dtype": image_dtype, "shape": image_shape, "names": image_names},
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ACTION_NAMES},
    }


def schema_markdown() -> str:
    state_rows = "\n".join(
        f"| {i} | {name} | {unit} |" for i, (name, unit) in enumerate(zip(STATE_NAMES, STATE_UNITS))
    )
    action_rows = "\n".join(
        f"| {i} | {name} | {unit} |" for i, (name, unit) in enumerate(zip(ACTION_NAMES, ACTION_UNITS))
    )
    return (
        "## State Schema\n\n"
        "| index | name | unit |\n|---:|---|---|\n"
        f"{state_rows}\n\n"
        "## Action Schema\n\n"
        "| index | name | unit |\n|---:|---|---|\n"
        f"{action_rows}\n\n"
        "Gripper convention: `state[12]` is current binary open/close state; "
        "`action[6]` is binary absolute target gripper position. TCP action "
        "dimensions 0..5 are deltas.\n"
    )

