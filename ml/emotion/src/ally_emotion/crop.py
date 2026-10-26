"""The face crop: one function for training and for the phone (technical plan T14; D-65 decision 3).

The detector's box is made square around its centre, side = longer box side x (1 + margin); the
square is shrunk to size x size by an exact area average, and any part of it outside the picture
counts as black (0). Grey is BT.601: 0.299 R + 0.587 G + 0.114 B. Rounding is floor(v + 0.5),
clipped to 0-255.

So that the Kotlin port gives the same pixels, the arithmetic order is fixed: a horizontal pass,
then a vertical pass, each summing input pixels in increasing index order in double precision,
with weight = (overlap length of the pixel with the output interval) / s on each axis, where
s = side / size and output interval j is [x0 + j*s, x0 + (j+1)*s). Pixels with no overlap add
exactly 0, so the port may skip them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_BT601 = (0.299, 0.587, 0.114)


@dataclass(frozen=True)
class Box:
    """Pixels, top-left origin."""

    x: float
    y: float
    w: float
    h: float


def square_box(box: Box, margin: float) -> Box:
    side = max(box.w, box.h) * (1 + margin)
    cx, cy = box.x + box.w / 2, box.y + box.h / 2
    return Box(cx - side / 2, cy - side / 2, side, side)


def to_grey(rgb: NDArray[np.uint8]) -> NDArray[np.float64]:
    r, g, b = (rgb[..., k].astype(np.float64) for k in range(3))
    return _BT601[0] * r + _BT601[1] * g + _BT601[2] * b


def area_resize(grey: NDArray[np.float64], size: int) -> NDArray[np.float64]:
    """Shrink (or grow) a square grey array to size x size by the exact area average."""
    h, w = grey.shape
    if h != w:
        raise ValueError(f"area_resize needs a square array, got {h} x {w}")
    return _area_sample(grey, Box(0, 0, w, h), size)


def crop_face(
    image: NDArray[np.uint8], box: Box, margin: float, size: int = 48
) -> NDArray[np.uint8]:
    """size x size uint8 grey face from an H x W x 3 RGB or H x W grey uint8 image."""
    if image.dtype != np.uint8:
        raise ValueError(f"image must be uint8, got {image.dtype}")
    if image.ndim == 3 and image.shape[2] == 3:
        grey = to_grey(image)
    elif image.ndim == 2:
        grey = image.astype(np.float64)
    else:
        raise ValueError(f"image must be H x W x 3 or H x W, got shape {image.shape}")
    if box.w <= 0 or box.h <= 0:
        raise ValueError(f"box must have a positive size, got {box}")
    values = _area_sample(grey, square_box(box, margin), size)
    return np.clip(np.floor(values + 0.5), 0, 255).astype(np.uint8)


def _area_sample(grey: NDArray[np.float64], square: Box, size: int) -> NDArray[np.float64]:
    s = square.w / size
    wx = _weights(square.x, s, size, grey.shape[1])  # size x W
    wy = _weights(square.y, s, size, grey.shape[0])  # size x H
    across = np.zeros((grey.shape[0], size))
    for c in range(grey.shape[1]):  # horizontal pass, input columns in increasing order
        across += grey[:, c : c + 1] * wx[:, c][np.newaxis, :]
    out = np.zeros((size, size))
    for r in range(grey.shape[0]):  # vertical pass, input rows in increasing order
        out += wy[:, r : r + 1] * across[r : r + 1, :]
    return out


def _weights(start: float, s: float, size: int, length: int) -> NDArray[np.float64]:
    """weights[j, c] = overlap of [start + j*s, start + (j+1)*s) with [c, c+1), divided by s."""
    w = np.zeros((size, length))
    for j in range(size):
        a, b = start + j * s, start + (j + 1) * s
        for c in range(max(0, math.floor(a)), min(length, math.ceil(b))):
            w[j, c] = (min(b, c + 1) - max(a, c)) / s
    return w
