"""Replay a built split (F1 `build.py`) through the live scorer (F3c Tasks 3 and 4, D-69).

This is how the live path is checked against the trained one (the scores of the saved
validation windows must reproduce the F3b numbers) and how the alert values are chosen (the
windows of each video in time order, as the camera would give them).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ally_fall.data import BuiltSplit, window_tensor, window_video
from ally_fall.live.scorer import Scorer


@dataclass(frozen=True)
class VideoWindows:
    """The windows of one video: indices into the split's windows, in time order."""

    video: int
    dataset: str
    subject: int
    index: NDArray[np.int64]


def score_split(
    split: BuiltSplit, scorer: Scorer, chunk: int = 2048
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Fall score and "most likely class is fall or fallen" for every window of the split."""
    scores: list[NDArray[np.float32]] = []
    top: list[NDArray[np.bool_]] = []
    total = len(split.window_end)
    for start in range(0, total, chunk):
        index = np.arange(start, min(start + chunk, total), dtype=np.int64)
        windows = window_tensor(split, index, scorer.window_ticks)
        scores.append(scorer.score_batch(windows))
        top.append(scorer.top_is_fall_batch(windows))
    if not scores:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.bool_)
    return np.concatenate(scores), np.concatenate(top)


def window_groups(split: BuiltSplit) -> list[VideoWindows]:
    """One entry per video that has windows, with its windows in time order."""
    video = window_video(split)
    groups: list[VideoWindows] = []
    for v in np.unique(video):
        index = np.flatnonzero(video == v)
        index = index[np.argsort(split.window_end[index], kind="stable")]
        groups.append(
            VideoWindows(
                video=int(v),
                dataset=str(split.dataset[v]),
                subject=int(split.subject[v]),
                index=index.astype(np.int64),
            )
        )
    return groups
