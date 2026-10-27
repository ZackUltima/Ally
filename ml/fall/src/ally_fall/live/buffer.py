"""The last N ticks as one model window, in the layout of `data.window_tensor` (F3c, D-69)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ally_fall.keypoints import POINTS
from ally_fall.windows import WINDOW_TICKS


class WindowBuffer:
    """Keeps the last `length` ticks; once full, every new tick gives a (3, length, 33) window.

    The three channels are x, y and visibility, then time, then points, exactly as
    `data.window_tensor` returns for one window.
    """

    def __init__(self, length: int = WINDOW_TICKS) -> None:
        if length < 1:
            raise ValueError(f"window length must be at least 1, got {length}")
        self._length = length
        self._ticks: list[NDArray[np.float32]] = []

    def reset(self) -> None:
        self._ticks = []

    def push(self, features: NDArray[np.float32]) -> NDArray[np.float32] | None:
        """Add one tick ([33, 3]); return the window, or None while fewer than `length` ticks."""
        if features.shape != (POINTS, 3):
            raise ValueError(
                f"expected a tick of shape {(POINTS, 3)} (33 points), got {features.shape}"
            )
        self._ticks.append(features.astype(np.float32, copy=True))
        if len(self._ticks) > self._length:
            self._ticks.pop(0)
        if len(self._ticks) < self._length:
            return None
        stacked = np.stack(self._ticks)  # (length, 33, 3)
        return np.ascontiguousarray(stacked.transpose(2, 0, 1))
