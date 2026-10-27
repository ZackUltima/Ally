"""Tests for the pose-result mapping (F3c Task 5, D-69). The MediaPipe landmarker itself is
created in `PoseDetector` and checked by hand (listed as an exclusion in the sprint review)."""

from types import SimpleNamespace

import numpy as np

from ally_fall.keypoints import POINTS
from ally_fall.live.pose import landmarks_to_arrays, next_timestamp_ms


def _result(points: list[SimpleNamespace] | None) -> SimpleNamespace:
    return SimpleNamespace(pose_landmarks=[points] if points else [])


def test_no_person_gives_zero_points_and_zero_visibility() -> None:
    xy, vis = landmarks_to_arrays(_result(None))
    assert xy.shape == (POINTS, 2) and vis.shape == (POINTS,)
    assert not xy.any() and not vis.any()


def test_points_are_mapped_in_order() -> None:
    points = [SimpleNamespace(x=i / 100, y=1 - i / 100, visibility=i / 40) for i in range(POINTS)]
    xy, vis = landmarks_to_arrays(_result(points))
    assert np.allclose(xy[:, 0], np.arange(POINTS) / 100)
    assert np.allclose(xy[:, 1], 1 - np.arange(POINTS) / 100)
    assert np.allclose(vis, np.arange(POINTS) / 40)
    assert xy.dtype == np.float32 and vis.dtype == np.float32


def test_a_missing_visibility_is_zero_and_extra_points_are_ignored() -> None:
    points = [SimpleNamespace(x=0.5, y=0.5, visibility=None) for _ in range(POINTS + 3)]
    xy, vis = landmarks_to_arrays(_result(points))
    assert xy.shape == (POINTS, 2) and not vis.any()


def test_timestamps_are_strictly_increasing_whole_milliseconds() -> None:
    assert next_timestamp_ms(0.0, -1) == 0
    assert next_timestamp_ms(0.0004, 0) == 1  # same millisecond: pushed on by one
    assert next_timestamp_ms(1.2345, 0) == 1234
    assert next_timestamp_ms(1.0, 1500) == 1501
