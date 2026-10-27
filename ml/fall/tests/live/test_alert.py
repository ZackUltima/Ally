"""Tests for the live alert rule (F3c Task 4, D-69)."""

import numpy as np
import pytest

from ally_fall.live.alert import AlertRule, active_mask


def _run(rule: AlertRule, scores: list[float]) -> tuple[list[bool], list[bool]]:
    states = [rule.update(s) for s in scores]
    return [s.fired for s in states], [s.active for s in states]


def test_below_the_threshold_never_fires() -> None:
    fired, active = _run(AlertRule(0.5, 3, 0), [0.1, 0.49, 0.2, 0.0] * 5)
    assert not any(fired) and not any(active)


def test_n_ticks_in_a_row_fire_once_and_stay_active() -> None:
    fired, active = _run(AlertRule(0.5, 3, 0), [0.9, 0.9, 0.9, 0.9, 0.9])
    assert fired == [False, False, True, False, False]
    assert active == [False, False, True, True, True]


def test_a_score_equal_to_the_threshold_counts() -> None:
    fired, _ = _run(AlertRule(0.5, 2, 0), [0.5, 0.5])
    assert fired == [False, True]


def test_a_dip_resets_the_count() -> None:
    fired, _ = _run(AlertRule(0.5, 3, 0), [0.9, 0.9, 0.1, 0.9, 0.9])
    assert not any(fired)
    fired, _ = _run(AlertRule(0.5, 3, 0), [0.9, 0.9, 0.1, 0.9, 0.9, 0.9])
    assert fired == [False, False, False, False, False, True]


def test_the_banner_goes_off_when_the_score_drops() -> None:
    _, active = _run(AlertRule(0.5, 2, 0), [0.9, 0.9, 0.9, 0.2, 0.9])
    assert active == [False, True, True, False, False]


def test_the_cool_down_blocks_a_second_fire_but_not_the_banner() -> None:
    rule = AlertRule(0.5, 2, 6)
    fired, active = _run(rule, [0.9, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9, 0.9])
    # first fire at index 1; cool-down of 6 updates covers indices 2-7; next fire at index 10
    assert [i for i, f in enumerate(fired) if f] == [1, 10]
    assert active[4] is True  # banner shown while the fire is blocked


def test_the_threshold_and_n_can_change_while_running() -> None:
    rule = AlertRule(0.9, 2, 0)
    assert not any(_run(rule, [0.7, 0.7])[0])
    rule.set_threshold(0.6)
    assert rule.threshold == 0.6
    assert _run(rule, [0.7, 0.7])[0] == [False, True]
    rule.set_required(1)
    assert rule.required == 1
    rule.update(0.1)
    assert rule.update(0.7).fired is True  # a run of one tick fires at once (cool-down is 0)


def test_values_are_checked() -> None:
    with pytest.raises(ValueError, match="threshold"):
        AlertRule(0.0, 3, 0)
    with pytest.raises(ValueError, match="threshold"):
        AlertRule(1.5, 3, 0)
    with pytest.raises(ValueError, match="required"):
        AlertRule(0.5, 0, 0)
    with pytest.raises(ValueError, match="cool"):
        AlertRule(0.5, 3, -1)
    rule = AlertRule(0.5, 3, 0)
    with pytest.raises(ValueError, match="threshold"):
        rule.set_threshold(2.0)
    with pytest.raises(ValueError, match="required"):
        rule.set_required(0)


def test_reset_clears_the_count_and_the_cool_down() -> None:
    rule = AlertRule(0.5, 2, 100)
    _run(rule, [0.9, 0.9])
    rule.reset()
    fired, _ = _run(rule, [0.9, 0.9])
    assert fired == [False, True]


def test_active_mask_equals_running_the_rule() -> None:
    scores = np.random.default_rng(0).uniform(0, 1, 500).astype(np.float32)
    rule = AlertRule(0.6, 4, 0)
    expected = np.array([rule.update(float(s)).active for s in scores])
    assert np.array_equal(active_mask(scores, 0.6, 4), expected)
