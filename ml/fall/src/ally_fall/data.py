"""Training data from a built split (F1 build.py): windows as (n, 3, 30, 33) arrays.

Windows are gathered from the stored ticks when needed, never stored repeated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ally_fall.augment import HIPS
from ally_fall.build import window_features
from ally_fall.windows import WINDOW_TICKS


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


def window_tensor(
    split: BuiltSplit, index: NDArray[np.int64], length: int = WINDOW_TICKS
) -> NDArray[np.float32]:
    """Windows `index` as (n, 3, length, 33): channels x, y, visibility; then time; then points.

    The default is the built 30-tick window; a shorter `length` is the last ticks of that window
    (the F3b window ablation), with the same label.
    """
    windows = window_features(split.features, split.window_end[index], length)  # (n, length, 33, 3)
    return np.ascontiguousarray(windows.transpose(0, 3, 1, 2))


def window_video(split: BuiltSplit) -> NDArray[np.int64]:
    """The video index of every window."""
    return (np.searchsorted(split.video_start, split.window_end, side="right") - 1).astype(np.int64)


def person_of(split: BuiltSplit) -> NDArray[np.int64]:
    """One id per window for its person: the (dataset, subject) pair of its video."""
    keys = [f"{d}/{s}" for d, s in zip(split.dataset.tolist(), split.subject.tolist(), strict=True)]
    _, video_person = np.unique(np.array(keys), return_inverse=True)
    return video_person[window_video(split)].astype(np.int64)


def hip_centred(windows: NDArray[np.float32]) -> NDArray[np.float32]:
    """Moves each window so its mean visible hip centre is (0, 0) (F3b ablation, D-55 item 8).

    Takes one window (3, 30, 33) or a batch (n, 3, 30, 33). The centre is the mean x and the
    mean y of the hip points over the ticks where they are visible, as `augment.scale` finds it.
    Missing points stay exactly (0, 0, 0); a window with no visible hip is left as it is. The
    result can be negative: it is not a picture coordinate, so nothing is marked missing.
    """
    if windows.ndim not in (3, 4) or windows.shape[-3] != 3:
        raise ValueError(f"expected a window or batch of shape (n, 3, ticks, points), got "
                         f"{windows.shape}")
    batch = windows if windows.ndim == 4 else windows[None]
    seen = batch[:, 2][:, :, HIPS] > 0  # (n, ticks, 2)
    count = seen.sum(axis=(1, 2))
    divisor = np.maximum(count, 1)
    centre = [(batch[:, axis][:, :, HIPS] * seen).sum(axis=(1, 2)) / divisor for axis in (0, 1)]
    out = batch.copy()
    for axis in (0, 1):
        shift = np.where(count > 0, centre[axis], 0.0).astype(np.float32)
        out[:, axis] = np.where(batch[:, 2] > 0, batch[:, axis] - shift[:, None, None],
                                batch[:, axis])
    return out if windows.ndim == 4 else out[0]
