"""Streaming version of the 10 fps resampling (technical plan §2, T2; offline: ticks.py).

Frames arrive one at a time. A tick every 0.1 s, starting at the first frame's time, takes the
keypoints of the latest frame at or before the tick, as on the phone and as `build.video_ticks`
does offline; a tick is given as soon as a frame at or after it has arrived. Points outside the
image become (0, 0) with visibility 0 (`ticks.mark_missing`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ally_fall.keypoints import POINTS
from ally_fall.ticks import _EPS, TICK_S, mark_missing


@dataclass(frozen=True)
class Tick:
    """One 10 fps tick: its time in seconds and its keypoints [33, 3] (x, y, visibility)."""

    time_s: float
    features: NDArray[np.float32]


class Ticker:
    """Turns frames (time, points, visibility) into ticks, in order."""

    def __init__(self) -> None:
        self._first_s: float | None = None
        self._count: int = 0  # ticks given so far
        self._last_s: float = 0.0
        self._latest: NDArray[np.float32] | None = None

    def reset(self) -> None:
        """Start a new stream: the next frame is the first frame (time may start over)."""
        self._first_s = None
        self._count = 0
        self._latest = None

    def _next_tick_s(self) -> float:
        assert self._first_s is not None
        return self._first_s + self._count * TICK_S

    def _emit(self) -> Tick:
        assert self._latest is not None
        tick = Tick(time_s=self._next_tick_s(), features=self._latest.copy())
        self._count += 1
        return tick

    def push(
        self, time_s: float, xy: NDArray[np.float32], vis: NDArray[np.float32]
    ) -> list[Tick]:
        """Add one frame; return the ticks now known (none, one, or several after a gap)."""
        if xy.shape != (POINTS, 2) or vis.shape != (POINTS,):
            raise ValueError(
                f"expected {POINTS} points as xy {(POINTS, 2)} and visibility {(POINTS,)}, "
                f"got {xy.shape} and {vis.shape}"
            )
        if self._first_s is not None and time_s <= self._last_s:
            raise ValueError(
                f"frame times must increase: {time_s:.4f} s after {self._last_s:.4f} s"
            )
        out: list[Tick] = []
        if self._first_s is None:
            self._first_s = time_s
        else:
            # Ticks before this frame belong to the previous frame.
            while self._next_tick_s() < time_s - _EPS:
                out.append(self._emit())
        marked_xy, marked_vis = mark_missing(xy, vis)
        self._latest = np.concatenate([marked_xy, marked_vis[:, None]], axis=-1).astype(np.float32)
        self._last_s = time_s
        # A tick at this frame's time (within rounding) takes this frame.
        while self._next_tick_s() <= time_s + _EPS:
            out.append(self._emit())
        return out
