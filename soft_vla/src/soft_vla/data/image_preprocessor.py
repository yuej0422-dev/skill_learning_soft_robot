from __future__ import annotations

import numpy as np


def ensure_rgb_uint8(image: np.ndarray, expected_shape: tuple[int, int, int] | None = None) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        raise ValueError(f"image dtype must be uint8, got {arr.dtype}.")
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"image must have shape HxWx3, got {arr.shape}.")
    if expected_shape is not None and tuple(arr.shape) != tuple(expected_shape):
        raise ValueError(f"image shape must be {expected_shape}, got {arr.shape}.")
    return arr


def images_are_distinct(images: list[np.ndarray]) -> bool:
    if len(images) < 2:
        return True
    first = np.asarray(images[0])
    return any(not np.array_equal(first, np.asarray(img)) for img in images[1:])

