"""30-tick windows with the last tick's label (D-55 item 2)."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

WINDOW_TICKS: int = 30


def make_windows(
    features: NDArray[np.float32],
    labels: NDArray[np.int64],
    length: int = WINDOW_TICKS,
    stride: int = 1,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Cut [T, K, 3] features into [N, length, K, 3] windows labelled by their last tick.

    Windows whose last tick is unlabelled (-1) are dropped; a video shorter than one
    window gives zero windows.
    """
    ticks, points, channels = features.shape
    if ticks < length:
        return (
            np.zeros((0, length, points, channels), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    view = sliding_window_view(features, length, axis=0)[::stride]  # [N, K, 3, length]
    windows = np.moveaxis(view, -1, 1)
    last = labels[length - 1 :: stride][: len(windows)]
    keep = last != -1
    return np.ascontiguousarray(windows[keep], dtype=np.float32), last[keep].astype(np.int64)
