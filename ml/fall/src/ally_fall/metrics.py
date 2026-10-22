"""Evaluation measures (technical plan §4; D-13, D-55; T5).

Per window: fall-class (fall or fallen) recall, precision, F1, specificity and the four
counts, plus F1 per label. Per fall event (T5): a labelled event is one unbroken run of
fall-class ticks (a fall and the fallen that follows it, or a person found fallen); it is
detected if the fall flow would start inside it. The flow starts when the prediction turns
on, at most once per `merge_s` seconds (§6: one alert per fall); a start outside every
event is a false alert. Intervals resample people, not windows (D-55).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ally_fall.ticks import TICK_S


def _ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


@dataclass(frozen=True)
class Binary:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def recall(self) -> float:
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def precision(self) -> float:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def f1(self) -> float:
        return _ratio(2 * self.tp, 2 * self.tp + self.fp + self.fn)

    @property
    def specificity(self) -> float:
        return _ratio(self.tn, self.tn + self.fp)

    def __add__(self, other: Binary) -> Binary:
        return Binary(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn,
                      self.tn + other.tn)


def binary(pred: NDArray[np.bool_], true: NDArray[np.bool_]) -> Binary:
    return Binary(tp=int((pred & true).sum()), fp=int((pred & ~true).sum()),
                  fn=int((~pred & true).sum()), tn=int((~pred & ~true).sum()))


def per_label_f1(pred: NDArray[np.int64], true: NDArray[np.int64], n_classes: int
                 ) -> NDArray[np.float64]:
    return np.array([binary(pred == c, true == c).f1 for c in range(n_classes)])


@dataclass(frozen=True)
class Events:
    detected: int
    missed: int
    false_alerts: int


def _runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """(first, last + 1) tick indices of every unbroken run of True."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist(),
                    strict=True))


def count_events(
    times_s: NDArray[np.float64],
    pred_fall: NDArray[np.bool_],
    true_labels: NDArray[np.int64],
    fall: int,
    fallen: int,
    merge_s: float = 30.0,
) -> Events:
    """Detected, missed and false-alert counts for one video (ticks in time order)."""
    events = [(times_s[a], times_s[b - 1] + TICK_S)
              for a, b in _runs(np.isin(true_labels, [fall, fallen]))]
    starts: list[float] = []
    for a, _ in _runs(pred_fall):
        if not starts or times_s[a] - starts[-1] >= merge_s:
            starts.append(float(times_s[a]))
    hit = [False] * len(events)
    false_alerts = 0
    for t in starts:
        inside = [i for i, (begin, end) in enumerate(events) if begin <= t < end]
        if inside:
            hit[inside[0]] = True
        else:
            false_alerts += 1
    return Events(detected=sum(hit), missed=len(events) - sum(hit), false_alerts=false_alerts)


def person_bootstrap(
    per_person: Sequence[Binary], stat: Callable[[Binary], float], n: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """95% percentile interval of `stat` over `n` resamples of whole people."""
    rng = np.random.default_rng(seed)
    people = list(per_person)
    values = []
    for _ in range(n):
        picked = rng.integers(len(people), size=len(people))
        total = Binary(0, 0, 0, 0)
        for i in picked:
            total = total + people[i]
        values.append(stat(total))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)
