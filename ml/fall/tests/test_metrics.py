import numpy as np
from numpy.typing import NDArray

from ally_fall.metrics import (
    Binary,
    Events,
    binary,
    count_events,
    per_label_f1,
    person_bootstrap,
)

FALL, FALLEN, WALK = 1, 2, 0


def test_binary_counts_and_measures() -> None:
    pred = np.array([1, 1, 1, 0, 0, 0, 0, 1, 0, 0], dtype=bool)
    true = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    b = binary(pred, true)
    assert (b.tp, b.fp, b.fn, b.tn) == (2, 2, 1, 5)
    assert abs(b.recall - 2 / 3) < 1e-9 and abs(b.precision - 0.5) < 1e-9
    assert abs(b.f1 - 4 / 7) < 1e-9 and abs(b.specificity - 5 / 7) < 1e-9


def test_binary_no_positive_predictions_gives_zero_precision() -> None:
    b = binary(np.zeros(4, dtype=bool), np.array([1, 0, 0, 0], dtype=bool))
    assert b.precision == 0.0 and b.f1 == 0.0 and b.recall == 0.0


def test_per_label_f1() -> None:
    f1 = per_label_f1(np.array([0, 1, 1, 2]), np.array([0, 1, 2, 2]), 4)
    np.testing.assert_allclose(f1, [1.0, 2 / 3, 2 / 3, 0.0])


def _timeline(spans: list[tuple[float, float, int]], end: float) -> tuple[NDArray, NDArray]:
    """Ticks every 0.1 s from 0 to `end`; label WALK except inside the given spans."""
    times = np.round(np.arange(0.0, end, 0.1), 1)
    labels = np.full(len(times), WALK)
    for start, stop, label in spans:
        labels[(times >= start) & (times < stop)] = label
    return times, labels


def _pred(times: NDArray, on: list[tuple[float, float]]) -> NDArray[np.bool_]:
    pred = np.zeros(len(times), dtype=bool)
    for start, stop in on:
        pred[(times >= start) & (times < stop)] = True
    return pred


def _events(times: NDArray, labels: NDArray, pred: NDArray) -> Events:
    return count_events(times, pred, labels, FALL, FALLEN)


def test_event_detected_within_fall_and_following_fallen() -> None:
    times, labels = _timeline([(10.0, 11.0, FALL), (11.0, 20.0, FALLEN)], 30.0)
    assert _events(times, labels, _pred(times, [(15.0, 15.5)])) == Events(1, 0, 0)


def test_event_start_outside_is_false_alert() -> None:
    times, labels = _timeline([(10.0, 11.0, FALL), (11.0, 20.0, FALLEN)], 50.0)
    assert _events(times, labels, _pred(times, [(40.0, 41.0)])) == Events(0, 1, 1)


def test_starts_within_30s_count_once() -> None:
    times, labels = _timeline([(10.0, 11.0, FALL), (11.0, 50.0, FALLEN)], 70.0)
    pred = _pred(times, [(10.5, 11.0), (12.0, 12.5), (35.0, 36.0), (60.0, 61.0)])
    assert _events(times, labels, pred) == Events(1, 0, 1)


def test_video_without_fall() -> None:
    times, labels = _timeline([], 20.0)
    assert _events(times, labels, _pred(times, [])) == Events(0, 0, 0)


def test_two_falls_each_counted_once() -> None:
    times, labels = _timeline([(5.0, 6.0, FALL), (6.0, 10.0, FALLEN),
                               (50.0, 51.0, FALL), (51.0, 55.0, FALLEN)], 60.0)
    pred = _pred(times, [(5.5, 9.0), (52.0, 53.0)])
    assert _events(times, labels, pred) == Events(2, 0, 0)


def test_bootstrap_resamples_people_not_windows() -> None:
    people = [Binary(tp=1000, fp=0, fn=0, tn=0), Binary(tp=0, fp=0, fn=10, tn=0)]
    low, high = person_bootstrap(people, lambda b: b.recall, n=2000)
    assert low < 0.05 and high > 0.95


def test_bootstrap_repeatable() -> None:
    people = [Binary(5, 1, 2, 9), Binary(3, 0, 4, 7), Binary(8, 2, 1, 6)]
    assert person_bootstrap(people, lambda b: b.f1, n=500, seed=4) == person_bootstrap(
        people, lambda b: b.f1, n=500, seed=4)
