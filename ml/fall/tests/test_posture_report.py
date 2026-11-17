"""Tests for the posture report on validation windows (P1 Task 3, D-72)."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ally_fall.data import BuiltSplit
from ally_fall.posture import POSTURES
from ally_fall.posture_report import (
    BAR,
    SeedResult,
    bar_met,
    confusion,
    missing_share,
    posture_scores,
    report_markdown,
    true_postures,
)

NAMES = (
    "walk", "fall", "fallen", "sit_down", "sitting", "lie_down", "lying", "stand_up",
    "standing", "other", "kneel_down", "kneeling", "squat_down", "squatting",
    "crawl", "jump",
)


def test_true_postures_come_from_the_label_names() -> None:
    labels = np.array([NAMES.index(n) for n in ("standing", "fallen", "sitting", "jump")])
    got = true_postures(labels, NAMES)
    assert [POSTURES[i] for i in got] == ["upright", "lying", "sitting", "other"]


def test_confusion_counts_true_rows_against_predicted_columns() -> None:
    true = np.array([0, 0, 1, 2, 2, 2], dtype=np.int64)
    pred = np.array([0, 1, 1, 2, 2, 0], dtype=np.int64)
    matrix = confusion(pred, true)
    assert matrix.shape == (5, 5)
    assert matrix[0].tolist() == [1, 1, 0, 0, 0]
    assert matrix[2].tolist() == [1, 0, 2, 0, 0]
    assert matrix.sum() == 6


def test_scores_are_per_posture_recall_precision_and_f1() -> None:
    true = np.array([0, 0, 0, 0, 1, 1], dtype=np.int64)
    pred = np.array([0, 0, 0, 1, 1, 0], dtype=np.int64)
    scores = {s.posture: s for s in posture_scores(pred, true)}
    upright = scores["upright"]
    assert upright.windows == 4
    assert upright.recall == pytest.approx(3 / 4)
    assert upright.precision == pytest.approx(3 / 4)
    assert upright.f1 == pytest.approx(3 / 4)
    assert scores["sitting"].f1 == pytest.approx(0.5)
    assert scores["changing"].windows == 0  # a posture with no windows scores 0, not an error
    assert scores["changing"].f1 == 0.0


def test_the_bar_needs_all_three_postures_on_every_seed() -> None:
    assert BAR == 0.80
    good = {"upright": 0.85, "sitting": 0.9, "lying": 0.81, "changing": 0.1, "other": 0.2}
    weak = {**good, "lying": 0.79}
    assert bar_met([good, good, good])
    assert not bar_met([good, good, weak])
    assert not bar_met([])


def test_missing_share_is_the_share_of_ticks_with_no_pose() -> None:
    features = np.ones((12, 33, 3), dtype=np.float32)
    features[3:6, :, 2] = 0.0  # ticks 3, 4, 5 have no visible point
    split = BuiltSplit(
        features=features,
        video_start=np.array([0], dtype=np.int64),
        video_len=np.array([12], dtype=np.int64),
        subject=np.array([1], dtype=np.int64),
        dataset=np.array(["d"]),
        window_end=np.array([5, 11], dtype=np.int64),
        y=np.array([0, 0], dtype=np.int64),
    )
    share = missing_share(split, ticks=6)
    assert share.tolist() == pytest.approx([0.5, 0.0])


def _result(seed: int, f1: float) -> SeedResult:
    n = len(POSTURES)
    matrix = np.eye(n, dtype=np.int64) * 10
    scores = posture_scores(
        np.repeat(np.arange(n), 10).astype(np.int64), np.repeat(np.arange(n), 10).astype(np.int64)
    )
    scores = [replace(s, f1=f1) for s in scores]
    return SeedResult(seed, matrix, scores, {"UP-Fall/2": f1}, {"UP-Fall": f1}, {"none": f1})


def test_the_report_states_validation_only_and_the_bar_result() -> None:
    text = report_markdown([_result(0, 0.9), _result(1, 0.9), _result(2, 0.7)])
    assert "validation" in text.lower()
    assert "test split" in text.lower()  # says it was not opened
    assert "missed" in text.lower()
    assert "upright" in text and "lying" in text
    assert "no older adults" in text.lower() or "older adults" in text.lower()
    met = report_markdown([_result(0, 0.9), _result(1, 0.9), _result(2, 0.9)])
    assert "met" in met.lower() and "missed" not in met.lower()


def test_main_refuses_paths_in_the_test_split(tmp_path: Path) -> None:
    from ally_fall.posture_report import main

    bad = tmp_path / "test" / "out.md"
    with pytest.raises(ValueError, match="only validation"):
        main(["--config-dir", str(tmp_path), "--data-root", str(tmp_path), "--out", str(bad)])


def test_a_met_pooled_bar_still_names_the_datasets_below_it() -> None:
    def seed(n: int) -> SeedResult:
        base = _result(n, 0.9)
        return replace(base, by_dataset={"up_fall": 0.95, "le2i": 0.55})

    text = report_markdown([seed(0), seed(1), seed(2)])
    assert "pooled" in text.lower()
    assert "le2i" in text.split("## F1 per posture")[0]  # named next to the verdict
    assert "up_fall" not in text.split("## F1 per posture")[0]
    clean = report_markdown(
        [replace(_result(i, 0.9), by_dataset={"up_fall": 0.95}) for i in range(3)]
    )
    assert "below the bar" not in clean.lower()


def test_the_report_says_when_each_dataset_is_one_person() -> None:
    one = report_markdown([_result(0, 0.9)])  # one person, one dataset in the helper
    assert "one person" in one.lower()
    many = replace(_result(0, 0.9), by_person={"a/1": 0.9, "a/2": 0.9})
    assert "one person" not in report_markdown([many]).lower()
