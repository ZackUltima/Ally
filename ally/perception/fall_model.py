"""Fall detectors over the per-frame features of `fall_features` (methods.md §6.1 steps 3–4).

Every detector has the same shape so the fusion layer and the Sprint 2 ablation (rule vs trees vs GRU) can
swap them: `reset()`, then `update(row, ts) -> score` per frame, where a score ≥ τ_fall is what the event
engine turns into `SUSPECTED_FALL`. Nothing here does I/O or reads a clock.

Implemented now: the rule-based reference detector (step 3a). The gradient-boosted-tree baseline and the
GRU (ONNX export) arrive in Sprint 2 alongside `scripts/train_fall.py`.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from ally.perception.fall_features import N_FEATURES, F


class FallDetector(Protocol):
    """Per-frame scorer. `update` returns a fall score in [0, 1] for the frame just seen."""

    def reset(self) -> None: ...

    def update(self, row: np.ndarray, ts: float) -> float: ...


class RuleFallDetector:
    """Plan §6.1 step 3(a): *angle > 60° and vertical velocity > v_thr, followed by low hip height for 2 s*.

    Three phases, driven only by the feature row and its timestamp:

    idle   → an **impact** frame (torso angle > `angle_deg` and downward hip speed > `v_thr`) arms the
             detector.
    armed  → the hip must reach `hip_height_max` (normalised height, 0 = bottom of frame) within `settle_s`
             of the impact; the first such frame starts the lying timer. If `settle_s` passes first the
             candidate expires (a lurch that never reached the floor).
    lying  → a valid frame with the hip still low `lying_s` after the timer started **fires** (`update`
             returns 1.0 once); a valid frame with the hip back above `hip_height_max` cancels (got up).
    Frames whose features are NaN neither confirm nor cancel; the timers keep running.

    `angle_deg` (60°), `lying_s` (2 s) and `settle_s` (the 1.5 s feature window) are the plan's numbers.
    `v_thr` (frame-heights per second, downward) and `hip_height_max` are deliberately required: the plan
    leaves them to the Sprint 2 evaluation on UR Fall (decisions.md, "Rule baseline parameters").
    """

    def __init__(
        self,
        *,
        v_thr: float,
        hip_height_max: float,
        angle_deg: float = 60.0,
        lying_s: float = 2.0,
        settle_s: float = 1.5,
    ) -> None:
        for name, value in (("v_thr", v_thr), ("hip_height_max", hip_height_max)):
            if value is None or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number chosen by the Sprint 2 evaluation")
        if lying_s <= 0 or settle_s <= 0:
            raise ValueError("lying_s and settle_s must be positive")
        self.v_thr = float(v_thr)
        self.hip_height_max = float(hip_height_max)
        self.angle_deg = float(angle_deg)
        self.lying_s = float(lying_s)
        self.settle_s = float(settle_s)
        self._impact_ts: float | None = None
        self._low_since: float | None = None

    @property
    def phase(self) -> str:
        if self._impact_ts is None:
            return "idle"
        return "lying" if self._low_since is not None else "armed"

    def reset(self) -> None:
        self._impact_ts = None
        self._low_since = None

    def update(self, row: np.ndarray, ts: float) -> float:
        r = np.asarray(row, dtype=np.float64)
        if r.shape != (N_FEATURES,):
            raise ValueError(f"row must have shape [{N_FEATURES}], got {r.shape}")
        angle, vy, hip_h = r[F["torso_angle_deg"]], r[F["hip_vy"]], r[F["hip_height"]]
        hip_valid = not np.isnan(hip_h)
        is_low = hip_valid and hip_h <= self.hip_height_max

        if self._impact_ts is not None and self._low_since is None:  # armed
            if is_low:
                self._low_since = float(ts)
            elif ts - self._impact_ts > self.settle_s:
                self.reset()  # never reached the floor in time

        if self._low_since is not None:  # lying
            if hip_valid and not is_low:
                self.reset()  # back up: cancelled
            elif is_low and ts - self._low_since >= self.lying_s:
                self.reset()  # one score per fall; the engine debounces from here
                return 1.0

        if self._impact_ts is None:
            is_impact = (
                (not np.isnan(angle)) and (not np.isnan(vy)) and angle > self.angle_deg and vy > self.v_thr
            )
            if is_impact:
                self._impact_ts = float(ts)
        return 0.0

    def score_sequence(self, features: np.ndarray, ts: np.ndarray) -> np.ndarray:
        """Offline convenience for evaluation: reset, then `update` over every row → scores [T]."""
        f = np.asarray(features, dtype=np.float64)
        t = np.asarray(ts, dtype=np.float64)
        if f.ndim != 2 or f.shape[0] != t.shape[0]:
            raise ValueError("features [T, F] and ts [T] must agree on T")
        self.reset()
        return np.array([self.update(f[i], float(t[i])) for i in range(f.shape[0])])


__all__ = ["FallDetector", "RuleFallDetector"]
