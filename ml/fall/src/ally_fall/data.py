"""Training data from a built split (F1 build.py): windows as (n, 3, 30, 33) arrays.

Windows are gathered from the stored ticks when needed, never stored repeated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ally_fall.build import window_features


@dataclass(frozen=True)
class BuiltSplit:
    """One built split; `subject` and `dataset` are per video, `window_end` and `y` per window."""

    features: NDArray[np.float32]
    video_start: NDArray[np.int64]
    video_len: NDArray[np.int64]
    subject: NDArray[np.int64]
    dataset: NDArray[np.str_]
    window_end: NDArray[np.int64]
    y: NDArray[np.int64]


def load_split(path: Path, label_map: NDArray[np.int64]) -> BuiltSplit:
    """Load built/<pose>/<split>.npz with each window's label mapped into the label set."""
    with np.load(path, allow_pickle=False) as data:
        return BuiltSplit(
            features=data["features"].astype(np.float32),
            video_start=data["video_start"].astype(np.int64),
            video_len=data["video_len"].astype(np.int64),
            subject=data["subject"].astype(np.int64),
            dataset=data["dataset"],
            window_end=data["window_end"].astype(np.int64),
            y=label_map[data["y"]].astype(np.int64),
        )


def window_tensor(split: BuiltSplit, index: NDArray[np.int64]) -> NDArray[np.float32]:
    """Windows `index` as (n, 3, 30, 33): channels x, y, visibility; then time; then points."""
    windows = window_features(split.features, split.window_end[index])  # (n, 30, 33, 3)
    return np.ascontiguousarray(windows.transpose(0, 3, 1, 2))


def window_video(split: BuiltSplit) -> NDArray[np.int64]:
    """The video index of every window."""
    return (np.searchsorted(split.video_start, split.window_end, side="right") - 1).astype(np.int64)


def person_of(split: BuiltSplit) -> NDArray[np.int64]:
    """One id per window for its person: the (dataset, subject) pair of its video."""
    keys = [f"{d}/{s}" for d, s in zip(split.dataset.tolist(), split.subject.tolist(), strict=True)]
    _, video_person = np.unique(np.array(keys), return_inverse=True)
    return video_person[window_video(split)].astype(np.int64)
