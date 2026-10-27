"""Tests for choosing the alert values on validation windows (F3c Task 4, D-69)."""

import numpy as np

from ally_fall.data import BuiltSplit
from ally_fall.keypoints import POINTS
from ally_fall.live.tune import GridRow, alert_grid, choose_default, grid_markdown
from ally_fall.windows import WINDOW_TICKS

FALL, FALLEN = 1, 2


def _split() -> tuple[BuiltSplit, np.ndarray]:
    """Two 80-tick videos; video 0 has one fall event; video 1 has none."""
    length = 80
    windows_per_video = length - WINDOW_TICKS + 1  # 51
    starts = np.array([0, length], dtype=np.int64)
    window_end = np.concatenate(
        [np.arange(s + WINDOW_TICKS - 1, s + length) for s in starts]
    ).astype(np.int64)
    y = np.zeros(len(window_end), dtype=np.int64)
    y[10:25] = FALL  # video 0, windows 10-24
    scores = np.full(len(window_end), 0.05, dtype=np.float32)
    scores[12:24] = 0.95  # a clear fall signal inside the event
    scores[windows_per_video + 30 : windows_per_video + 32] = 0.95  # 2-tick spike, no fall
    split = BuiltSplit(
        features=np.zeros((2 * length, POINTS, 3), dtype=np.float32),
        video_start=starts,
        video_len=np.array([length, length], dtype=np.int64),
        subject=np.array([1, 2], dtype=np.int64),
        dataset=np.array(["a", "b"]),
        window_end=window_end,
        y=y,
    )
    return split, scores


def test_a_longer_run_removes_the_short_spike_but_keeps_the_fall() -> None:
    split, scores = _split()
    rows = {
        (r.threshold, r.required): r
        for r in alert_grid(split, scores, (FALL, FALLEN), [0.5, 0.99], [1, 3])
    }
    assert rows[(0.5, 1)].detected == 1 and rows[(0.5, 1)].false_alerts == 1  # the spike alarms
    assert rows[(0.5, 3)].detected == 1 and rows[(0.5, 3)].false_alerts == 0
    assert rows[(0.99, 1)].detected == 0 and rows[(0.99, 1)].missed == 1  # never reaches 0.99


def test_false_alerts_per_hour_uses_the_window_time() -> None:
    split, scores = _split()
    row = alert_grid(split, scores, (FALL, FALLEN), [0.5], [1])[0]
    hours = 2 * 51 * 0.1 / 3600.0
    assert abs(row.false_per_hour - 1 / hours) < 1e-6


def test_the_default_keeps_the_falls_then_minimises_false_alerts() -> None:
    rows = [
        GridRow(0.5, 1, 57, 0, 30, 3.0),
        GridRow(0.5, 3, 57, 0, 12, 1.2),
        GridRow(0.7, 3, 56, 1, 4, 0.4),
        GridRow(0.9, 5, 40, 17, 0, 0.0),
    ]
    # best detected = 57; those within 1 fall: (0.5,1), (0.5,3), (0.7,3); fewest false alerts wins
    assert choose_default(rows) == (0.7, 3)


def test_ties_prefer_the_smaller_run_then_the_lower_threshold() -> None:
    rows = [GridRow(0.6, 5, 57, 0, 2, 0.2), GridRow(0.6, 3, 57, 0, 2, 0.2),
            GridRow(0.5, 3, 57, 0, 2, 0.2)]
    assert choose_default(rows) == (0.5, 3)


def test_the_table_lists_every_row_and_marks_the_default() -> None:
    split, scores = _split()
    rows = alert_grid(split, scores, (FALL, FALLEN), [0.5, 0.9], [1, 3])
    text = grid_markdown(rows, "run-x", (0.5, 3), hours=0.05)
    assert "run-x" in text and "validation" in text.lower()
    body = [
        line for line in text.splitlines() if line.startswith("| ") and "0." in line.split("|")[1]
    ]
    assert len(body) == 4
    assert sum("**0.50**" in line for line in body) == 1  # only the chosen row is in bold
