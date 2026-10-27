"""Frames in, fall score and alert state out (F3c Task 5, D-69).

Camera frames are not kept here (D-26): the pipeline takes the pose of each frame and keeps only
keypoints of the last `HISTORY_TICKS` ticks (10 s), which the PO may save with a key press.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ally_fall.keypoints import POINTS
from ally_fall.live.alert import AlertRule, AlertState
from ally_fall.live.buffer import WindowBuffer
from ally_fall.live.scorer import Scorer
from ally_fall.live.ticker import Ticker

HISTORY_TICKS: int = 100  # 10 s at 10 fps


@dataclass(frozen=True)
class FrameResult:
    """What one frame produced: the pose, the ticks it completed, and the latest score."""

    time_s: float
    xy: NDArray[np.float32]
    vis: NDArray[np.float32]
    new_ticks: int
    score: float | None  # score of the newest tick scored by this frame, else None
    state: AlertState | None


class LivePipeline:
    def __init__(self, scorer: Scorer, rule: AlertRule) -> None:
        self.rule = rule
        self._scorer = scorer
        self._ticker = Ticker()
        self._buffer = WindowBuffer(scorer.window_ticks)
        self._history: deque[NDArray[np.float32]] = deque(maxlen=HISTORY_TICKS)
        self.last_score: float | None = None
        self.active: bool = False
        self.fired_total: int = 0

    def reset(self) -> None:
        self._ticker.reset()
        self._buffer.reset()
        self._history.clear()
        self.rule.reset()
        self.last_score = None
        self.active = False
        self.fired_total = 0

    def process(
        self, time_s: float, xy: NDArray[np.float32], vis: NDArray[np.float32]
    ) -> FrameResult:
        """Add one frame's pose (xy [33, 2], visibility [33]) taken at `time_s` seconds."""
        score: float | None = None
        state: AlertState | None = None
        ticks = self._ticker.push(time_s, xy, vis)
        for tick in ticks:
            self._history.append(tick.features)
            window = self._buffer.push(tick.features)
            if window is None:
                continue
            score = self._scorer.score(window)
            state = self.rule.update(score)
            self.last_score = score
            self.active = state.active
            self.fired_total += int(state.fired)
        return FrameResult(time_s, xy, vis, len(ticks), score, state)

    def recent_keypoints(self) -> NDArray[np.float32]:
        """Keypoints [n, 33, 3] of the last ticks (at most 10 s), oldest first; no images."""
        if not self._history:
            return np.zeros((0, POINTS, 3), dtype=np.float32)
        return np.stack(self._history)
