import math

import numpy as np
from numpy.typing import NDArray

from ally_fall.baseline import (
    Thresholds,
    fit_from_features,
    fit_thresholds,
    hip_drop_speed,
    predict,
    rule_features,
    torso_angle,
)

RULE = Thresholds(speed=1.0, angle=1.0)


def _body(hip_y: NDArray[np.float64], angle: NDArray[np.float64], torso: float = 0.2) -> NDArray:
    """A (3, 30, 33) window with only shoulders and hips; angle 0 = upright, pi/2 = flat."""
    w = np.zeros((3, 30, 33), dtype=np.float32)
    for point in (23, 24):
        w[0, :, point], w[1, :, point], w[2, :, point] = 0.5, hip_y, 1.0
    for point in (11, 12):
        w[0, :, point] = 0.5 + torso * np.sin(angle)
        w[1, :, point] = hip_y - torso * np.cos(angle)
        w[2, :, point] = 1.0
    return w


def _ramp(start: float, end: float, ticks: int) -> NDArray[np.float64]:
    """Hold `start`, then move linearly to `end` over the last `ticks` ticks."""
    values = np.full(30, start)
    values[30 - ticks:] = np.linspace(start, end, ticks + 1)[1:]
    return values


def test_standing_still_is_not_fall() -> None:
    w = _body(np.full(30, 0.6), np.zeros(30))[None]
    assert hip_drop_speed(w)[0] < 1e-6 and torso_angle(w)[0] < 1e-6
    assert not predict(w, RULE)[0]


def test_fast_drop_to_horizontal_is_fall() -> None:
    w = _body(_ramp(0.4, 0.6, 5), _ramp(0.0, math.pi / 2, 5))[None]  # 1 torso length in 0.5 s
    assert abs(hip_drop_speed(w)[0] - 2.0) < 1e-3
    assert abs(torso_angle(w)[0] - math.pi / 2) < 1e-3
    assert predict(w, RULE)[0]


def test_slow_lie_down_is_not_fall_by_speed() -> None:
    w = _body(_ramp(0.3, 0.4, 30), _ramp(0.0, math.pi / 2, 30))[None]  # hips stay in the upper half
    assert hip_drop_speed(w)[0] < 1.0
    assert not predict(w, RULE)[0]


def test_missing_hips_predicts_not_fall() -> None:
    w = _body(_ramp(0.4, 0.6, 5), _ramp(0.0, math.pi / 2, 5))
    w[:, :, 23:25] = 0.0
    w = w[None]
    speed, angle = hip_drop_speed(w), torso_angle(w)
    assert np.isfinite(speed).all() and np.isfinite(angle).all()
    assert not predict(w, RULE)[0]


def test_fit_thresholds_recovers_separating_values() -> None:
    flat = _ramp(0.0, math.pi / 2, 30)
    torso = 0.05
    # speed s torso lengths/s on every tick (30 steps of 0.1 s); the hips stay in the upper
    # half, so only the speed part of the rule can tell the two groups apart
    windows = [_body(_ramp(0.05, 0.05 + s * 3.0 * torso, 30), flat, torso)
               for s in (2.6, 2.8, 3.0, 1.0, 1.2, 1.4)]
    is_fall = np.array([True, True, True, False, False, False])
    t = fit_thresholds(np.stack(windows), is_fall, np.arange(0.5, 5.01, 0.25),
                       np.arange(0.3, 1.41, 0.1))
    assert 1.4 < t.speed <= 2.6
    assert predict(np.stack(windows), t).tolist() == is_fall.tolist()


def test_fit_from_chunked_features_matches_fit_on_windows() -> None:
    flat = _ramp(0.0, math.pi / 2, 30)
    windows = np.stack([_body(_ramp(0.05, 0.05 + s * 0.15, 30), flat, 0.05)
                        for s in (2.6, 1.0, 3.0, 1.4)])
    is_fall = np.array([True, False, True, False])
    grid = (np.arange(0.5, 5.01, 0.25), np.arange(0.3, 1.41, 0.1))
    chunks = [rule_features(windows[:2]), rule_features(windows[2:])]
    features = tuple(np.concatenate(parts) for parts in zip(*chunks, strict=True))
    assert fit_from_features(*features, is_fall, *grid) == fit_thresholds(windows, is_fall, *grid)
