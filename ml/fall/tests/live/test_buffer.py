"""Tests for the 30-tick window buffer (F3c Task 2, D-69)."""

import numpy as np
import pytest

from ally_fall.build import window_features
from ally_fall.keypoints import POINTS
from ally_fall.live.buffer import WindowBuffer
from ally_fall.windows import WINDOW_TICKS


def _features(count: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0, 1, size=(count, POINTS, 3)).astype(np.float32)


def test_nothing_until_the_window_is_full() -> None:
    buffer = WindowBuffer()
    features = _features(WINDOW_TICKS)
    for tick in features[:-1]:
        assert buffer.push(tick) is None
    window = buffer.push(features[-1])
    assert window is not None
    assert window.shape == (3, WINDOW_TICKS, POINTS)


def test_layout_is_channels_then_time_then_points() -> None:
    buffer = WindowBuffer()
    features = _features(WINDOW_TICKS)
    window = None
    for tick in features:
        window = buffer.push(tick)
    assert window is not None
    assert np.array_equal(window[0], features[:, :, 0])  # x
    assert np.array_equal(window[1], features[:, :, 1])  # y
    assert np.array_equal(window[2], features[:, :, 2])  # visibility


def test_windows_equal_the_offline_windows_at_every_tick() -> None:
    features = _features(75, seed=2)
    buffer = WindowBuffer()
    live = [w for w in (buffer.push(tick) for tick in features) if w is not None]
    ends = np.arange(WINDOW_TICKS - 1, len(features), dtype=np.int64)
    offline = window_features(features, ends).transpose(0, 3, 1, 2)  # as data.window_tensor
    assert len(live) == len(offline)
    assert np.array_equal(np.stack(live), offline)


def test_the_window_slides_by_one_tick() -> None:
    features = _features(WINDOW_TICKS + 1)
    buffer = WindowBuffer()
    windows = [buffer.push(tick) for tick in features]
    assert windows[-1] is not None and windows[-2] is not None
    assert np.array_equal(windows[-1][:, :-1], windows[-2][:, 1:])


def test_a_returned_window_is_not_changed_by_later_ticks() -> None:
    features = _features(WINDOW_TICKS + 1)
    buffer = WindowBuffer()
    first = None
    for tick in features[:WINDOW_TICKS]:
        first = buffer.push(tick)
    assert first is not None
    kept = first.copy()
    buffer.push(features[-1])
    assert np.array_equal(first, kept)


def test_shorter_length_for_the_window_ablation() -> None:
    buffer = WindowBuffer(length=21)
    window = None
    for tick in _features(21):
        window = buffer.push(tick)
    assert window is not None and window.shape == (3, 21, POINTS)


def test_reset_empties_the_buffer() -> None:
    buffer = WindowBuffer()
    for tick in _features(WINDOW_TICKS):
        buffer.push(tick)
    buffer.reset()
    assert buffer.push(_features(1)[0]) is None


def test_wrong_shape_is_refused() -> None:
    with pytest.raises(ValueError, match="33"):
        WindowBuffer().push(np.zeros((POINTS, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="length"):
        WindowBuffer(length=0)
