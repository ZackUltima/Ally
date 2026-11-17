"""The posture report end to end on a small made-up split and a random-weight scorer (P1, D-72)."""

from pathlib import Path

import numpy as np
from live_helpers import NAMES, make_scorer

from ally_fall.data import BuiltSplit
from ally_fall.keypoints import POINTS
from ally_fall.posture import POSTURES
from ally_fall.posture_report import evaluate, predict_postures, report_markdown


def _split() -> BuiltSplit:
    rng = np.random.default_rng(0)
    features = rng.uniform(0.0, 1.0, size=(80, POINTS, 3)).astype(np.float32)
    features[40:50, :, 2] = 0.0  # ten ticks with no pose
    labels = [NAMES.index(n) for n in ("standing", "sitting", "fallen", "sit_down", "jump")]
    return BuiltSplit(
        features=features,
        video_start=np.array([0, 40], dtype=np.int64),
        video_len=np.array([40, 40], dtype=np.int64),
        subject=np.array([3, 4], dtype=np.int64),
        dataset=np.array(["up_fall", "le2i"]),
        window_end=np.array([35, 38, 70, 75, 79], dtype=np.int64),
        y=np.array(labels, dtype=np.int64),
    )


def test_every_window_gets_one_predicted_posture(tmp_path: Path) -> None:
    scorer = make_scorer(tmp_path)
    pred = predict_postures(scorer, _split(), chunk=2)
    assert pred.shape == (5,)
    assert set(pred.tolist()) <= set(range(len(POSTURES)))
    assert np.array_equal(pred, predict_postures(scorer, _split(), chunk=100))


def test_evaluate_counts_every_window_once_in_each_view(tmp_path: Path) -> None:
    result = evaluate(1, make_scorer(tmp_path), _split(), NAMES)
    assert result.seed == 1
    assert int(result.matrix.sum()) == 5
    assert [s.windows for s in result.scores] == [1, 1, 1, 1, 1]  # one window of each posture
    assert set(result.by_dataset) <= {"up_fall", "le2i"}
    assert set(result.by_person) <= {"up_fall/3", "le2i/4"}
    assert set(result.by_missing) <= {
        "no ticks lost", "up to a fifth of the ticks", "a fifth to half of the ticks",
        "over half of the ticks",
    }
    assert "validation" in report_markdown([result]).lower()
