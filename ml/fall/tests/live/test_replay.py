"""Tests for replaying a built split through the live scorer (F3c Tasks 3 and 4, D-69)."""

from pathlib import Path

import numpy as np
from live_helpers import make_scorer

from ally_fall.data import BuiltSplit, window_tensor
from ally_fall.keypoints import POINTS
from ally_fall.live.replay import score_split, window_groups
from ally_fall.windows import WINDOW_TICKS


def _split() -> BuiltSplit:
    rng = np.random.default_rng(1)
    lengths = np.array([45, 38], dtype=np.int64)
    starts = np.array([0, 45], dtype=np.int64)
    features = rng.uniform(0, 1, size=(int(lengths.sum()), POINTS, 3)).astype(np.float32)
    ends: list[int] = []
    for start, length in zip(starts, lengths, strict=True):
        ends.extend(range(int(start) + WINDOW_TICKS - 1, int(start + length)))
    window_end = np.array(ends, dtype=np.int64)
    return BuiltSplit(
        features=features,
        video_start=starts,
        video_len=lengths,
        subject=np.array([1, 2], dtype=np.int64),
        dataset=np.array(["a", "b"]),
        window_end=window_end,
        y=np.zeros(len(window_end), dtype=np.int64),
    )


def test_scores_equal_scoring_the_windows_directly(tmp_path: Path) -> None:
    split, scorer = _split(), make_scorer(tmp_path)
    scores, top_fall = score_split(split, scorer, chunk=7)
    direct = scorer.score_batch(window_tensor(split, np.arange(len(split.window_end))))
    assert np.allclose(scores, direct, atol=1e-6)
    assert top_fall.shape == scores.shape and top_fall.dtype == np.bool_


def test_the_chunk_size_does_not_change_the_result(tmp_path: Path) -> None:
    split, scorer = _split(), make_scorer(tmp_path)
    a, fa = score_split(split, scorer, chunk=5)
    b, fb = score_split(split, scorer, chunk=1000)
    assert np.allclose(a, b, atol=1e-6) and np.array_equal(fa, fb)


def test_windows_are_grouped_by_video_in_time_order() -> None:
    split = _split()
    groups = window_groups(split)
    assert [g.video for g in groups] == [0, 1]
    assert [len(g.index) for g in groups] == [45 - WINDOW_TICKS + 1, 38 - WINDOW_TICKS + 1]
    for group in groups:
        assert np.all(np.diff(split.window_end[group.index]) == 1)
    assert groups[0].dataset == "a" and groups[1].subject == 2
