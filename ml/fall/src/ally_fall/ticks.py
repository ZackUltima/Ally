"""Resampling to the phone's 10 fps grid (technical plan §2, T2, D-55 item 1).

Keypoints at each 0.1 s tick come from the latest frame at or before the tick, as on
the phone; labels come from the nearest frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

TICK_S: float = 0.1
_EPS: float = 1e-6


@dataclass(frozen=True)
class Segment:
    """One labelled stretch of a video, in seconds; the end is exclusive."""

    start_s: float
    end_s: float
    label: int


def mark_missing(
    xy: NDArray[np.float32], vis: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Points outside the image (or NaN) become (0, 0) with visibility 0."""
    outside = np.isnan(xy).any(axis=-1) | (xy < 0.0).any(axis=-1) | (xy > 1.0).any(axis=-1)
    out_xy = np.where(outside[..., None], np.float32(0.0), xy).astype(np.float32)
    out_vis = np.where(outside, np.float32(0.0), vis).astype(np.float32)
    return out_xy, out_vis


def tick_times(frame_times_s: NDArray[np.float64]) -> NDArray[np.float64]:
    """Ticks every 0.1 s from the first frame's time up to the last frame's time."""
    first, last = float(frame_times_s[0]), float(frame_times_s[-1])
    count = int(np.floor((last - first) / TICK_S + _EPS)) + 1
    return first + np.arange(count, dtype=np.float64) * TICK_S


def latest_frame_index(
    frame_times_s: NDArray[np.float64], ticks: NDArray[np.float64]
) -> NDArray[np.int64]:
    """For each tick, the index of the last frame with time <= tick."""
    index = np.searchsorted(frame_times_s, ticks + _EPS, side="right") - 1
    return np.clip(index, 0, len(frame_times_s) - 1).astype(np.int64)


def nearest_frame_index(
    frame_times_s: NDArray[np.float64], ticks: NDArray[np.float64]
) -> NDArray[np.int64]:
    """For each tick, the index of the nearest frame; a tie goes to the earlier frame."""
    after = np.clip(np.searchsorted(frame_times_s, ticks, side="left"), 0, len(frame_times_s) - 1)
    before = np.clip(after - 1, 0, len(frame_times_s) - 1)
    earlier_is_closer = (ticks - frame_times_s[before]) <= (frame_times_s[after] - ticks)
    return np.where(earlier_is_closer, before, after).astype(np.int64)


def frame_labels(frame_times_s: NDArray[np.float64], segments: list[Segment]) -> NDArray[np.int64]:
    """Label of the segment holding each frame (start <= t < end), or -1 in a gap."""
    labels = np.full(len(frame_times_s), -1, dtype=np.int64)
    for segment in segments:
        inside = (frame_times_s >= segment.start_s) & (frame_times_s < segment.end_s)
        labels[inside] = segment.label
    return labels
