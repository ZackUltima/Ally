"""Tests for the live pipeline: frames in, ticks, windows, scores and alert state out
(F3c Task 5, D-69). Nothing from the camera is stored; only keypoints of the last ticks."""

from pathlib import Path

import numpy as np
from live_helpers import make_scorer

from ally_fall.build import window_features
from ally_fall.keypoints import POINTS
from ally_fall.live.alert import AlertRule
from ally_fall.live.pipeline import HISTORY_TICKS, LivePipeline
from ally_fall.live.ticker import Ticker
from ally_fall.windows import WINDOW_TICKS


def _frames(seconds: float, fps: float = 30.0, seed: int = 0) -> list[tuple[float, np.ndarray]]:
    rng = np.random.default_rng(seed)
    count = int(seconds * fps)
    return [
        (i / fps, rng.uniform(0.1, 0.9, size=(POINTS, 2)).astype(np.float32)) for i in range(count)
    ]


def _run(pipeline: LivePipeline, frames: list[tuple[float, np.ndarray]]) -> list:
    vis = np.full(POINTS, 0.9, dtype=np.float32)
    return [pipeline.process(t, xy, vis) for t, xy in frames]


def test_no_score_until_a_full_window_then_one_per_tick(tmp_path: Path) -> None:
    pipeline = LivePipeline(make_scorer(tmp_path), AlertRule(0.5, 3, 0))
    results = _run(pipeline, _frames(5.0))
    scored = [r for r in results if r.score is not None]
    assert results[0].score is None and results[60].score is None
    assert scored and scored[0].new_ticks == 1
    first_scored_at = results.index(scored[0])
    # ticks are at 0.0, 0.1, ...; the 30th tick (index 29) is at 2.9 s
    assert abs(results[first_scored_at].time_s - 0.1 * (WINDOW_TICKS - 1)) < 0.05
    assert sum(r.new_ticks for r in results) >= 49  # about 10 ticks a second for 5 s


def test_scores_equal_the_offline_window_scores(tmp_path: Path) -> None:
    scorer = make_scorer(tmp_path)
    pipeline = LivePipeline(scorer, AlertRule(0.5, 3, 0))
    frames = _frames(6.0, fps=24.0, seed=3)
    results = _run(pipeline, frames)
    live = [r.score for r in results for _ in range(r.new_ticks) if r.score is not None]
    # offline: the same ticks through the ticker alone, then windows, then the scorer
    ticker, features = Ticker(), []
    vis = np.full(POINTS, 0.9, dtype=np.float32)
    for t, xy in frames:
        features.extend(tick.features for tick in ticker.push(t, xy, vis))
    stack = np.stack(features)
    ends = np.arange(WINDOW_TICKS - 1, len(stack), dtype=np.int64)
    windows = window_features(stack, ends).transpose(0, 3, 1, 2)
    expected = scorer.score_batch(np.ascontiguousarray(windows))
    assert len(live) == len(expected)
    assert np.allclose(live, expected, atol=1e-6)


def test_the_alert_fires_once_when_the_score_holds(tmp_path: Path) -> None:
    pipeline = LivePipeline(make_scorer(tmp_path), AlertRule(1e-9, 3, 1000))
    results = _run(pipeline, _frames(6.0))
    assert pipeline.fired_total == 1
    assert sum(r.state.fired for r in results if r.state is not None) == 1
    assert pipeline.active


def test_the_history_keeps_only_the_last_ticks_as_keypoints(tmp_path: Path) -> None:
    pipeline = LivePipeline(make_scorer(tmp_path), AlertRule(0.5, 3, 0))
    _run(pipeline, _frames(25.0))
    recent = pipeline.recent_keypoints()
    assert recent.shape == (HISTORY_TICKS, POINTS, 3)


def test_reset_starts_over(tmp_path: Path) -> None:
    pipeline = LivePipeline(make_scorer(tmp_path), AlertRule(1e-9, 1, 0))
    _run(pipeline, _frames(5.0))
    pipeline.reset()
    assert pipeline.last_score is None and pipeline.fired_total == 0 and not pipeline.active
    assert pipeline.recent_keypoints().shape[0] == 0
    xy = np.full((POINTS, 2), 0.5, np.float32)
    first = pipeline.process(0.0, xy, np.ones(POINTS, np.float32))
    assert first.score is None and first.new_ticks == 1  # time may start again from zero
