"""Tests for 30-tick windows (D-55 item 2)."""

import numpy as np

from ally_fall.windows import WINDOW_TICKS, make_windows

K = 33


def _features(ticks: int) -> np.ndarray:
    # Each tick's values equal its index, so a window's position can be read back.
    return np.repeat(np.arange(ticks, dtype=np.float32), K * 3).reshape(ticks, K, 3)


def test_window_length_is_30() -> None:
    assert WINDOW_TICKS == 30


def test_windows_shape_and_label() -> None:
    labels = np.arange(40, dtype=np.int64)
    x, y = make_windows(_features(40), labels)
    assert x.shape == (11, 30, K, 3)
    assert y.tolist() == list(range(29, 40))
    assert x[3, 0, 0, 0] == 3 and x[3, -1, 0, 0] == 32


def test_short_video() -> None:
    x, y = make_windows(_features(29), np.zeros(29, dtype=np.int64))
    assert x.shape == (0, 30, K, 3) and y.shape == (0,)


def test_unlabelled_last_tick_dropped() -> None:
    labels = np.zeros(40, dtype=np.int64)
    labels[35] = -1
    x, y = make_windows(_features(40), labels)
    assert len(y) == 10
    assert 35 not in [int(window[-1, 0, 0]) for window in x]


def test_windows_do_not_share_memory_with_input() -> None:
    features = _features(31)
    x, _ = make_windows(features, np.zeros(31, dtype=np.int64))
    features[0, 0, 0] = 99
    assert x[0, 0, 0, 0] == 0
