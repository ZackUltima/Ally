"""The hand-set threshold rule our fall model must beat (D-37 criterion 1; technical plan §3).

Fall class (fall or fallen) when the hips drop fast and the torso ends near horizontal,
or when the torso is near horizontal with the hips in the lower half of the picture (the
fallen case). Speeds are in torso lengths per second, so the rule does not depend on how
far the person is from the camera. The two thresholds are fitted on training windows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ally_fall.ticks import TICK_S

Windows = NDArray[np.float32]  # (n, 3, 30, 33)
SHOULDERS: tuple[int, int] = (11, 12)
HIPS: tuple[int, int] = (23, 24)


@dataclass(frozen=True)
class Thresholds:
    speed: float
    angle: float


def _mid(w: Windows, pair: tuple[int, int]) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Mean (x, y) of the visible points of a pair per tick: (n, 30, 2), and whether any is seen."""
    vis = w[:, 2][:, :, list(pair)] > 0  # (n, 30, 2)
    count = vis.sum(axis=-1)
    xy = np.stack([w[:, 0][:, :, list(pair)], w[:, 1][:, :, list(pair)]], axis=-1)
    total = (xy * vis[..., None]).sum(axis=2)
    seen = count > 0
    return total / np.maximum(count, 1)[..., None], seen


def _torso(w: Windows) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Mid-hip (n, 30, 2), shoulder-to-hip vector (n, 30, 2), and ticks where both are seen."""
    hip, hip_seen = _mid(w, HIPS)
    shoulder, shoulder_seen = _mid(w, SHOULDERS)
    return hip, hip - shoulder, hip_seen & shoulder_seen


def _torso_length(vector: NDArray[np.float64], both: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Median shoulder-to-hip distance over the ticks where both are seen; 0 if none."""
    lengths = np.linalg.norm(vector, axis=-1)
    return np.array([float(np.median(row[seen])) if seen.any() else 0.0
                     for row, seen in zip(lengths, both, strict=True)])


def hip_drop_speed(w: Windows) -> NDArray[np.float32]:
    """Largest downward speed of the mid-hip between visible ticks, in torso lengths per second."""
    hip, vector, both = _torso(w)
    _, hip_seen = _mid(w, HIPS)
    torso = _torso_length(vector, both)
    speeds = np.zeros(len(w), dtype=np.float32)
    for i in range(len(w)):
        ticks = np.flatnonzero(hip_seen[i])
        if len(ticks) < 2 or torso[i] <= 0:
            continue
        drop = np.diff(hip[i, ticks, 1]) / (np.diff(ticks) * TICK_S)
        speeds[i] = max(float(drop.max()), 0.0) / torso[i]
    return speeds


def torso_angle(w: Windows) -> NDArray[np.float32]:
    """Angle of the shoulder-to-hip line from vertical at the last tick both are seen (radians)."""
    _, vector, both = _torso(w)
    angles = np.zeros(len(w), dtype=np.float32)
    for i in range(len(w)):
        ticks = np.flatnonzero(both[i])
        if len(ticks):
            dx, dy = vector[i, ticks[-1]]
            angles[i] = float(np.arctan2(abs(dx), abs(dy)))
    return angles


def _hips_low(w: Windows) -> NDArray[np.bool_]:
    """Whether the last visible mid-hip is in the lower half of the picture."""
    hip, seen = _mid(w, HIPS)
    low = np.zeros(len(w), dtype=bool)
    for i in range(len(w)):
        ticks = np.flatnonzero(seen[i])
        low[i] = bool(len(ticks)) and hip[i, ticks[-1], 1] > 0.5
    return low


def _rule(
    speed: NDArray[np.float32], angle: NDArray[np.float32], low: NDArray[np.bool_], t: Thresholds
) -> NDArray[np.bool_]:
    flat = angle >= t.angle
    return (flat & (speed >= t.speed)) | (flat & low)


RuleFeatures = tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]


def rule_features(w: Windows) -> RuleFeatures:
    """(hip drop speed, torso angle, hips low) per window; computed in chunks for large splits."""
    return hip_drop_speed(w), torso_angle(w), _hips_low(w)


def predict_from_features(features: RuleFeatures, t: Thresholds) -> NDArray[np.bool_]:
    return _rule(*features, t)


def predict(w: Windows, t: Thresholds) -> NDArray[np.bool_]:
    """Fall class (fall or fallen) per window."""
    return predict_from_features(rule_features(w), t)


def fit_thresholds(
    w: Windows, is_fall: NDArray[np.bool_], speeds: Sequence[float], angles: Sequence[float]
) -> Thresholds:
    """The grid pair with the best fall-class F1 on these windows (the first one on a tie)."""
    return fit_from_features(*rule_features(w), is_fall, speeds, angles)


def fit_from_features(
    speed: NDArray[np.float32],
    angle: NDArray[np.float32],
    low: NDArray[np.bool_],
    is_fall: NDArray[np.bool_],
    speeds: Sequence[float],
    angles: Sequence[float],
) -> Thresholds:
    best, best_f1 = Thresholds(float(speeds[0]), float(angles[0])), -1.0
    for s in speeds:
        for a in angles:
            t = Thresholds(float(s), float(a))
            pred = _rule(speed, angle, low, t)
            tp = int((pred & is_fall).sum())
            f1 = 2 * tp / max(int(pred.sum()) + int(is_fall.sum()), 1)
            if f1 > best_f1:
                best, best_f1 = t, f1
    return best
