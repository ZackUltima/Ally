"""RuleFallDetector: the §6.1 step 3(a) reference rule on hand-built feature rows.

The test thresholds (v_thr, hip_height_max) are inputs to exercise the mechanism, not the project's numbers —
those are chosen in Sprint 2 on UR Fall (decisions.md).
"""

from __future__ import annotations

import numpy as np
import pytest

from ally.perception.fall_features import N_FEATURES, F
from ally.perception.fall_model import FallDetector, RuleFallDetector

FPS = 15.0
V_THR = 0.5  # frame-heights / s (test input)
HIP_MAX = 0.3  # normalised hip height (test input)


def row(angle: float = 0.0, vy: float = 0.0, hip_h: float = 0.6) -> np.ndarray:
    r = np.zeros(N_FEATURES)
    r[F["torso_angle_deg"]] = angle
    r[F["hip_vy"]] = vy
    r[F["hip_height"]] = hip_h
    r[F["bbox_aspect"]] = 0.5
    r[F["mean_conf"]] = 0.9
    return r


def nan_row() -> np.ndarray:
    return np.full(N_FEATURES, np.nan)


def detector(**kw) -> RuleFallDetector:
    return RuleFallDetector(v_thr=V_THR, hip_height_max=HIP_MAX, **kw)


def run(det: RuleFallDetector, rows: list[np.ndarray], fps: float = FPS) -> list[tuple[float, float]]:
    """[(ts, score)] for every frame."""
    return [(i / fps, det.update(r, i / fps)) for i, r in enumerate(rows)]


def fall_rows(n_lying: int) -> list[np.ndarray]:
    """5 upright frames, 3 impact frames (tilting fast, hip still high), then n_lying low frames."""
    upright = [row()] * 5
    impact = [
        row(angle=70.0, vy=1.2, hip_h=0.5),
        row(angle=80.0, vy=1.0, hip_h=0.4),
        row(angle=85.0, vy=0.8, hip_h=0.35),
    ]
    lying = [row(angle=88.0, vy=0.0, hip_h=0.1)] * n_lying
    return upright + impact + lying


# ---- construction -------------------------------------------------------------------------------


def test_protocol_conformance():
    det: FallDetector = detector()
    det.reset()
    assert det.update(row(), 0.0) == 0.0


def test_documented_defaults():
    det = detector()
    assert det.angle_deg == 60.0 and det.lying_s == 2.0 and det.settle_s == 1.5


@pytest.mark.parametrize("kw", [{"v_thr": float("nan")}, {"hip_height_max": float("inf")}, {"v_thr": None}])
def test_undecided_thresholds_are_rejected(kw):
    args = {"v_thr": V_THR, "hip_height_max": HIP_MAX, **kw}
    with pytest.raises(ValueError, match="Sprint 2"):
        RuleFallDetector(**args)


def test_thresholds_are_keyword_only_and_required():
    with pytest.raises(TypeError):
        RuleFallDetector()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        RuleFallDetector(0.5, 0.3)  # type: ignore[misc]


def test_bad_row_shape_raises():
    with pytest.raises(ValueError, match="row"):
        detector().update(np.zeros(N_FEATURES + 1), 0.0)


# ---- the rule -------------------------------------------------------------------------------------


def test_fall_fires_once_two_seconds_after_reaching_the_floor():
    det = detector()
    out = run(det, fall_rows(n_lying=60))
    fired = [ts for ts, s in out if s == 1.0]
    assert len(fired) == 1
    t_low = 8 / FPS  # first low-hip frame (index 8)
    assert fired[0] == pytest.approx(t_low + 2.0, abs=1 / FPS)
    assert det.phase == "idle"


def test_score_sequence_matches_online_updates():
    rows = fall_rows(n_lying=60)
    ts = np.arange(len(rows)) / FPS
    det = detector()
    online = np.array([s for _, s in run(det, rows)])
    assert np.array_equal(det.score_sequence(np.stack(rows), ts), online)


def test_lying_shorter_than_two_seconds_does_not_fire():
    out = run(detector(), fall_rows(n_lying=25))  # 25 frames = 1.67 s
    assert all(s == 0.0 for _, s in out)


def test_getting_back_up_cancels():
    det = detector()
    rows = fall_rows(n_lying=15) + [row(angle=20.0, vy=-0.5, hip_h=0.6)] + [row(angle=88.0, hip_h=0.1)] * 60
    out = run(det, rows)
    assert all(s == 0.0 for _, s in out)  # lying again afterwards needs a fresh impact


def test_slow_sit_down_is_not_a_fall():
    # torso tilts past 60° but the hip descends slowly (below v_thr); then stays low for a long time
    rows = [row()] * 5 + [row(angle=70.0, vy=0.2, hip_h=0.4)] * 5 + [row(angle=70.0, vy=0.0, hip_h=0.2)] * 60
    assert all(s == 0.0 for _, s in run(detector(), rows))


def test_fast_drop_without_tilt_is_not_a_fall():
    # e.g. crouching quickly: high downward speed but the torso stays near vertical
    rows = [row()] * 5 + [row(angle=20.0, vy=1.5, hip_h=0.4)] * 3 + [row(angle=20.0, hip_h=0.1)] * 60
    assert all(s == 0.0 for _, s in run(detector(), rows))


def test_lying_from_the_start_without_impact_never_fires():
    rows = [row(angle=88.0, hip_h=0.1)] * 90
    assert all(s == 0.0 for _, s in run(detector(), rows))


def test_lurch_that_never_reaches_the_floor_expires():
    det = detector()
    rows = [row()] * 5 + [row(angle=70.0, vy=1.2, hip_h=0.6)] + [row(angle=30.0, hip_h=0.6)] * 30
    out = run(det, rows)
    assert all(s == 0.0 for _, s in out)
    assert det.phase == "idle"
    # settling low only *after* settle_s passed does not count either
    more = run(det, [row(angle=88.0, hip_h=0.1)] * 60)
    assert all(s == 0.0 for _, s in more)


def test_phase_progression():
    det = detector()
    rows = fall_rows(n_lying=60)
    phases = []
    for i, r in enumerate(rows):
        det.update(r, i / FPS)
        phases.append(det.phase)
    assert phases[4] == "idle"
    assert phases[5] == "armed"  # first impact frame
    assert phases[8] == "lying"  # first low-hip frame
    assert phases[-1] == "idle"  # fired and reset


# ---- missing data -----------------------------------------------------------------------------------


def test_nan_gap_while_lying_neither_cancels_nor_confirms():
    det = detector()
    rows = fall_rows(n_lying=10) + [nan_row()] * 20 + [row(angle=88.0, hip_h=0.1)] * 10
    out = run(det, rows)
    fired = [ts for ts, s in out if s == 1.0]
    assert len(fired) == 1
    assert fired[0] == pytest.approx(8 / FPS + 2.0, abs=1 / FPS)


def test_all_nan_never_fires():
    assert all(s == 0.0 for _, s in run(detector(), [nan_row()] * 100))


def test_nan_angle_or_velocity_is_not_an_impact():
    r = row(angle=80.0, vy=2.0, hip_h=0.5)
    r[F["torso_angle_deg"]] = np.nan
    det = detector()
    det.update(r, 0.0)
    assert det.phase == "idle"
    r = row(angle=80.0, vy=2.0, hip_h=0.5)
    r[F["hip_vy"]] = np.nan
    det.update(r, 0.1)
    assert det.phase == "idle"


# ---- reset / reuse -----------------------------------------------------------------------------------


def test_reset_clears_an_armed_candidate():
    det = detector()
    det.update(row(angle=80.0, vy=2.0, hip_h=0.5), 0.0)
    assert det.phase == "armed"
    det.reset()
    assert det.phase == "idle"
    assert all(s == 0.0 for _, s in run(det, [row(angle=88.0, hip_h=0.1)] * 60))


def test_two_falls_in_a_row_fire_twice():
    det = detector()
    rows = fall_rows(n_lying=40) + [row()] * 5 + fall_rows(n_lying=40)
    assert sum(s for _, s in run(det, rows)) == 2.0
