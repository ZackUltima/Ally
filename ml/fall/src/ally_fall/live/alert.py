"""The live alert rule (F3c Task 4, D-69; decision 3 of the plan).

One fall score arrives per 10 fps tick. The alert is active while the score has been at or
above the threshold on `required` ticks in a row (one dip resets the count); it fires once when
it becomes active, and not again until `cool_down` ticks have passed. The banner follows
`active`; counted alerts follow `fired`. This is the model's decision only: the Kotlin fall
flow (check-in, escalation; F3a) is not reproduced here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class AlertState:
    fired: bool  # the alert started on this tick and was not blocked by the cool-down
    active: bool  # the score has held for `required` ticks and has not dipped since
    run: int  # ticks in a row at or above the threshold


def _check_threshold(value: float) -> None:
    if not 0.0 < value <= 1.0:
        raise ValueError(f"threshold must be above 0 and at most 1, got {value}")


def _check_required(value: int) -> None:
    if value < 1:
        raise ValueError(f"required ticks must be at least 1, got {value}")


class AlertRule:
    def __init__(self, threshold: float, required: int, cool_down: int) -> None:
        _check_threshold(threshold)
        _check_required(required)
        if cool_down < 0:
            raise ValueError(f"cool-down must be 0 or more ticks, got {cool_down}")
        self._threshold = threshold
        self._required = required
        self._cool_down = cool_down
        self._run = 0
        self._blocked_for = 0

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def required(self) -> int:
        return self._required

    def set_threshold(self, value: float) -> None:
        _check_threshold(value)
        self._threshold = value

    def set_required(self, value: int) -> None:
        _check_required(value)
        self._required = value

    def reset(self) -> None:
        self._run = 0
        self._blocked_for = 0

    def update(self, score: float) -> AlertState:
        """Feed one tick's fall score (0 to 1)."""
        self._run = self._run + 1 if score >= self._threshold else 0
        active = self._run >= self._required
        fired = active and self._run == self._required and self._blocked_for == 0
        if fired:
            self._blocked_for = self._cool_down + 1  # this tick plus `cool_down` more
        if self._blocked_for > 0:
            self._blocked_for -= 1
        return AlertState(fired=fired, active=active, run=self._run)


def active_mask(
    scores: NDArray[np.float32], threshold: float, required: int
) -> NDArray[np.bool_]:
    """`active` for every tick of one video's scores, in time order (no cool-down needed)."""
    rule = AlertRule(threshold, required, 0)
    return np.array([rule.update(float(s)).active for s in scores], dtype=np.bool_)
