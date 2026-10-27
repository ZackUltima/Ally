"""Tests for the streaming 10 fps ticker (F3c Task 2, D-69).

The key test feeds made-up frames through the live ticker and through the offline build
(`build.video_ticks`) and requires identical features, so the live path cannot drift from the
trained one.
"""

import numpy as np
import pytest
from numpy.typing import NDArray

from ally_fall.build import video_ticks
from ally_fall.keypoints import POINTS, KeypointSeq
from ally_fall.live.ticker import Tick, Ticker
from ally_fall.ticks import Segment


def _frames(
    times: NDArray[np.float64], seed: int = 0
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Made-up raw points: some outside the image or NaN, visibilities between 0 and 1."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-0.2, 1.2, size=(len(times), POINTS, 2)).astype(np.float32)
    xy[rng.random((len(times), POINTS)) < 0.05] = np.nan
    vis = rng.uniform(0.0, 1.0, size=(len(times), POINTS)).astype(np.float32)
    return xy, vis


def _live(
    times: NDArray[np.float64], xy: NDArray[np.float32], vis: NDArray[np.float32]
) -> list[Tick]:
    ticker = Ticker()
    out: list[Tick] = []
    for time_s, frame_xy, frame_vis in zip(times, xy, vis, strict=True):
        out.extend(ticker.push(float(time_s), frame_xy, frame_vis))
    return out


def _offline(
    times: NDArray[np.float64], xy: NDArray[np.float32], vis: NDArray[np.float32]
) -> NDArray[np.float32]:
    seq = KeypointSeq("test", "v", 1, 1, "lite", times, xy, vis)
    features, _ = video_ticks(seq, [Segment(0.0, 1e9, 0)])
    return features


@pytest.mark.parametrize("fps", [10.0, 15.0, 24.0, 30.0])
def test_live_equals_offline_at_common_frame_rates(fps: float) -> None:
    times = 1.7 + np.arange(int(fps * 6)) / fps
    xy, vis = _frames(times)
    ticks = _live(times, xy, vis)
    expected = _offline(times, xy, vis)
    assert len(ticks) == len(expected)
    assert np.array_equal(np.stack([t.features for t in ticks]), expected)


def test_live_equals_offline_with_jitter_and_a_gap() -> None:
    rng = np.random.default_rng(3)
    times = np.cumsum(rng.uniform(0.02, 0.08, size=120))
    times = np.concatenate([times, times[-1] + 0.9 + np.cumsum(rng.uniform(0.02, 0.08, size=60))])
    xy, vis = _frames(times, seed=4)
    ticks = _live(times, xy, vis)
    expected = _offline(times, xy, vis)
    assert len(ticks) == len(expected)
    assert np.array_equal(np.stack([t.features for t in ticks]), expected)


def test_tick_times_are_every_tenth_of_a_second_from_the_first_frame() -> None:
    times = 5.0 + np.arange(60) / 30.0
    xy, vis = _frames(times)
    ticks = _live(times, xy, vis)
    assert ticks[0].time_s == pytest.approx(5.0)
    assert np.allclose(np.diff([t.time_s for t in ticks]), 0.1)


def test_a_tick_waits_for_a_later_frame_when_none_has_arrived_yet() -> None:
    ticker = Ticker()
    xy, vis = _frames(np.zeros(1))
    assert len(ticker.push(0.0, xy[0], vis[0])) == 1  # the first frame is the first tick
    assert ticker.push(0.04, xy[0], vis[0]) == []  # next tick is at 0.1: not yet
    assert ticker.push(0.08, xy[0], vis[0]) == []
    assert len(ticker.push(0.12, xy[0], vis[0])) == 1  # now the 0.1 tick can be given


def test_a_gap_repeats_the_last_frame_for_the_missed_ticks() -> None:
    ticker = Ticker()
    xy, vis = _frames(np.zeros(3))
    ticker.push(0.0, xy[0], vis[0])
    ticks = ticker.push(0.55, xy[1], vis[1])  # ticks 0.1 .. 0.5 use frame 0, as on the phone
    assert [round(t.time_s, 1) for t in ticks] == [0.1, 0.2, 0.3, 0.4, 0.5]
    expected = _mark(xy[0], vis[0])
    assert all(np.array_equal(t.features, expected) for t in ticks)


def _mark(xy: NDArray[np.float32], vis: NDArray[np.float32]) -> NDArray[np.float32]:
    seq = KeypointSeq("t", "v", 1, 1, "lite", np.array([0.0, 0.1]), np.stack([xy, xy]),
                      np.stack([vis, vis]))
    features, _ = video_ticks(seq, [Segment(0.0, 1e9, 0)])
    return features[0]


def test_time_must_increase() -> None:
    ticker = Ticker()
    xy, vis = _frames(np.zeros(2))
    ticker.push(1.0, xy[0], vis[0])
    with pytest.raises(ValueError, match="increase"):
        ticker.push(1.0, xy[1], vis[1])


def test_shapes_are_checked() -> None:
    ticker = Ticker()
    with pytest.raises(ValueError, match="33"):
        ticker.push(0.0, np.zeros((5, 2), dtype=np.float32), np.zeros(5, dtype=np.float32))


def test_a_frame_with_no_person_gives_missing_points() -> None:
    ticker = Ticker()
    ticks = ticker.push(0.0, np.zeros((POINTS, 2), np.float32), np.zeros(POINTS, np.float32))
    assert ticks[0].features.shape == (POINTS, 3)
    assert not ticks[0].features[:, 2].any()


def test_reset_starts_a_new_stream() -> None:
    ticker = Ticker()
    xy, vis = _frames(np.zeros(2))
    ticker.push(10.0, xy[0], vis[0])
    ticker.reset()
    ticks = ticker.push(0.0, xy[1], vis[1])  # time may restart after a reset
    assert len(ticks) == 1 and ticks[0].time_s == 0.0
