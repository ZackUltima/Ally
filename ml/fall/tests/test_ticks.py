"""Tests for resampling to 10 fps ticks (technical plan §2, T2, D-55)."""

import numpy as np

from ally_fall.ticks import (
    Segment,
    frame_labels,
    latest_frame_index,
    mark_missing,
    nearest_frame_index,
    tick_times,
)


def test_mark_missing() -> None:
    xy = np.array([[[1.2, 0.5], [0.5, -0.1], [np.nan, 0.5], [0.5, 0.5]]], dtype=np.float32)
    vis = np.array([[0.9, 0.9, 0.9, 0.9]], dtype=np.float32)
    out_xy, out_vis = mark_missing(xy, vis)
    assert np.array_equal(out_xy[0, :3], np.zeros((3, 2), dtype=np.float32))
    assert np.array_equal(out_vis[0, :3], np.zeros(3, dtype=np.float32))
    assert np.allclose(out_xy[0, 3], [0.5, 0.5])
    assert out_vis[0, 3] == np.float32(0.9)


def test_mark_missing_does_not_change_inputs() -> None:
    xy = np.array([[[1.2, 0.5]]], dtype=np.float32)
    vis = np.array([[0.9]], dtype=np.float32)
    mark_missing(xy, vis)
    assert xy[0, 0, 0] == np.float32(1.2)


def test_ticks_30fps() -> None:
    frames = np.arange(90) / 30.0
    ticks = tick_times(frames)
    assert len(ticks) == 30
    assert np.isclose(ticks[0], 0.0) and np.isclose(ticks[-1], 2.9)
    assert np.array_equal(latest_frame_index(frames, ticks), np.arange(0, 90, 3))


def test_ticks_start_at_first_frame() -> None:
    frames = 5.0 + np.arange(10) / 10.0
    ticks = tick_times(frames)
    assert np.isclose(ticks[0], 5.0) and len(ticks) == 10


def test_ticks_25fps_latest_not_nearest() -> None:
    frames = np.arange(50) * 0.04
    ticks = tick_times(frames)
    # The tick at 0.1 s uses frame 2 (0.08 s), the latest at or before it, not frame 3 (0.12 s).
    assert latest_frame_index(frames, ticks)[1] == 2


def test_nearest_frame_differs_from_latest() -> None:
    frames = np.array([0.0, 0.04, 0.06, 0.11])
    ticks = np.array([0.1])
    assert nearest_frame_index(frames, ticks)[0] == 3
    assert latest_frame_index(frames, ticks)[0] == 2


def test_nearest_frame_tie_goes_to_earlier() -> None:
    frames = np.array([0.0, 0.2])
    assert nearest_frame_index(frames, np.array([0.1]))[0] == 0


def test_frame_labels_gap() -> None:
    segments = [Segment(0.0, 1.0, 0), Segment(1.5, 2.0, 1)]
    frames = np.array([0.5, 1.0, 1.2, 1.5, 2.5])
    assert frame_labels(frames, segments).tolist() == [0, -1, -1, 1, -1]
