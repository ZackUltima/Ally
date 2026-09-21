"""fall_features: per-frame kinematics on synthetic COCO-17 stick figures (no pose model needed)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from ally.contracts import KeypointFrame
from ally.perception.fall_features import (
    COCO17,
    FEATURE_NAMES,
    N_FEATURES,
    N_KPTS_COCO,
    WINDOW_FEATURE_NAMES,
    F,
    FeatureExtractor,
    frame_features,
    window_stats,
)

H, W = 480, 640
FPS = 15.0


def skeleton(
    hip_xy: tuple[float, float],
    angle_deg: float = 0.0,
    torso: float = 100.0,
    half_width: float = 20.0,
    conf: float = 0.9,
) -> np.ndarray:
    """[17, 3] stick figure. angle_deg = torso tilt from vertical (0 upright, 90 lying to the +x side)."""
    cx, cy = hip_xy
    a = np.radians(angle_deg)
    up = np.array([-np.sin(a), -np.cos(a)])  # image coords: y grows downward
    side = np.array([np.cos(a), -np.sin(a)])
    hips = np.array([cx, cy])
    shoulders = hips + torso * up
    head = shoulders + 0.4 * torso * up
    knees = hips - 0.5 * torso * up
    ankles = hips - 1.0 * torso * up
    kp = np.zeros((N_KPTS_COCO, 3))
    pts = {
        "nose": head,
        "l_eye": head + 5 * side,
        "r_eye": head - 5 * side,
        "l_ear": head + 10 * side,
        "r_ear": head - 10 * side,
        "l_shoulder": shoulders + half_width * side,
        "r_shoulder": shoulders - half_width * side,
        "l_elbow": shoulders + 1.5 * half_width * side - 0.3 * torso * up,
        "r_elbow": shoulders - 1.5 * half_width * side - 0.3 * torso * up,
        "l_wrist": shoulders + 1.5 * half_width * side - 0.6 * torso * up,
        "r_wrist": shoulders - 1.5 * half_width * side - 0.6 * torso * up,
        "l_hip": hips + 0.7 * half_width * side,
        "r_hip": hips - 0.7 * half_width * side,
        "l_knee": knees + 0.7 * half_width * side,
        "r_knee": knees - 0.7 * half_width * side,
        "l_ankle": ankles + 0.7 * half_width * side,
        "r_ankle": ankles - 0.7 * half_width * side,
    }
    for name, xy in pts.items():
        kp[COCO17[name], :2] = xy
        kp[COCO17[name], 2] = conf
    return kp


def sequence(hip_ys: list[float], angles: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    angles = angles or [0.0] * len(hip_ys)
    kps = np.stack([skeleton((320.0, y), a) for y, a in zip(hip_ys, angles, strict=True)])
    ts = np.arange(len(hip_ys)) / FPS
    return kps, ts


# ---- shape / naming ------------------------------------------------------------------------------


def test_feature_names_and_shapes():
    kps, ts = sequence([300.0] * 5)
    out = frame_features(kps, ts, (H, W))
    assert out.shape == (5, N_FEATURES)
    assert len(FEATURE_NAMES) == N_FEATURES
    assert set(F) == set(FEATURE_NAMES)
    assert len(WINDOW_FEATURE_NAMES) == 4 * N_FEATURES
    assert WINDOW_FEATURE_NAMES[:4] == (
        "torso_angle_deg_min",
        "torso_angle_deg_max",
        "torso_angle_deg_mean",
        "torso_angle_deg_std",
    )


@pytest.mark.parametrize(
    "bad",
    [np.zeros((5, 17)), np.zeros((5, 17, 2)), np.zeros((17, 3))],
)
def test_bad_keypoint_shape_raises(bad):
    with pytest.raises(ValueError, match="keypoints"):
        frame_features(bad, np.arange(bad.shape[0]) / FPS if bad.ndim == 3 else np.zeros(5), (H, W))


def test_non_increasing_ts_raises():
    kps, _ = sequence([300.0] * 3)
    with pytest.raises(ValueError, match="increasing"):
        frame_features(kps, np.array([0.0, 0.1, 0.1]), (H, W))


def test_ts_length_mismatch_raises():
    kps, _ = sequence([300.0] * 3)
    with pytest.raises(ValueError, match="ts"):
        frame_features(kps, np.array([0.0, 0.1]), (H, W))


# ---- static postures -----------------------------------------------------------------------------


def test_upright_person():
    kps, ts = sequence([300.0] * 4)
    out = frame_features(kps, ts, (H, W))
    assert np.allclose(out[:, F["torso_angle_deg"]], 0.0, atol=1e-6)
    assert np.allclose(out[:, F["hip_height"]], 1 - 300 / H)
    assert np.all(out[:, F["bbox_aspect"]] < 1.0)
    assert np.allclose(out[:, F["hip_vy"]], 0.0)
    assert np.allclose(out[:, F["hip_ay"]], 0.0)
    assert np.allclose(out[:, F["mean_conf"]], 0.9)


def test_lying_person():
    kps, ts = sequence([420.0] * 4, [90.0] * 4)
    out = frame_features(kps, ts, (H, W))
    assert np.allclose(out[:, F["torso_angle_deg"]], 90.0, atol=1e-6)
    assert np.all(out[:, F["bbox_aspect"]] > 1.0)
    assert np.all(out[:, F["hip_height"]] < 0.2)


def test_head_below_hips_exceeds_90():
    kps, ts = sequence([200.0] * 2, [150.0] * 2)
    out = frame_features(kps, ts, (H, W))
    assert np.allclose(out[:, F["torso_angle_deg"]], 150.0, atol=1e-6)


def test_angle_is_symmetric_in_tilt_direction():
    left, ts = sequence([300.0] * 2, [-45.0] * 2)
    right, _ = sequence([300.0] * 2, [45.0] * 2)
    assert np.allclose(frame_features(left, ts, (H, W))[:, 0], frame_features(right, ts, (H, W))[:, 0])


# ---- motion ---------------------------------------------------------------------------------------


def test_constant_downward_velocity_is_positive_and_normalised():
    rate_px_s = 240.0  # half a frame height per second
    ys = [100.0 + rate_px_s * i / FPS for i in range(10)]
    kps, ts = sequence(ys)
    out = frame_features(kps, ts, (H, W))
    assert np.allclose(out[:, F["hip_vy"]], rate_px_s / H)
    assert np.allclose(out[:, F["hip_ay"]], 0.0, atol=1e-9)
    assert np.all(np.diff(out[:, F["hip_height"]]) < 0)


def test_constant_acceleration():
    a_px_s2 = 960.0  # two frame heights per second squared
    ys = [100.0 + 0.5 * a_px_s2 * (i / FPS) ** 2 for i in range(12)]
    kps, ts = sequence(ys)
    out = frame_features(kps, ts, (H, W))
    # velocity is one-sided at both edges, so acceleration is exact only two rows in from each end
    assert np.allclose(out[2:-2, F["hip_ay"]], a_px_s2 / H, rtol=1e-6)


def test_irregular_timestamps_use_real_dt():
    ys = [100.0, 110.0, 130.0]
    kps, _ = sequence(ys)
    ts = np.array([0.0, 0.1, 0.3])  # second gap is twice as long
    out = frame_features(kps, ts, (H, W))
    assert np.isclose(out[0, F["hip_vy"]], (10.0 / H) / 0.1)
    assert np.isclose(out[2, F["hip_vy"]], (20.0 / H) / 0.2)


def test_single_frame_has_nan_derivatives_but_static_features():
    kps, ts = sequence([300.0])
    out = frame_features(kps, ts, (H, W))
    assert np.isnan(out[0, F["hip_vy"]]) and np.isnan(out[0, F["hip_ay"]])
    assert np.isclose(out[0, F["torso_angle_deg"]], 0.0)


# ---- missing data -----------------------------------------------------------------------------------


def test_missing_shoulders_give_nan_angle_but_keep_hip():
    kps, ts = sequence([300.0] * 3)
    kps[:, [COCO17["l_shoulder"], COCO17["r_shoulder"]], 2] = 0.0
    out = frame_features(kps, ts, (H, W))
    assert np.all(np.isnan(out[:, F["torso_angle_deg"]]))
    assert np.allclose(out[:, F["hip_height"]], 1 - 300 / H)


def test_one_shoulder_missing_uses_the_other():
    kps, ts = sequence([300.0] * 2)
    kps[:, COCO17["l_shoulder"], 2] = 0.0
    out = frame_features(kps, ts, (H, W))
    # remaining shoulder is offset sideways by half_width=20 over torso=100 → atan(0.2)
    assert np.allclose(out[:, F["torso_angle_deg"]], np.degrees(np.arctan(0.2)), atol=1e-6)


def test_missing_hips_give_nan_height_and_velocity_neighbours():
    kps, ts = sequence([300.0] * 5)
    kps[2, [COCO17["l_hip"], COCO17["r_hip"]], 2] = 0.0
    out = frame_features(kps, ts, (H, W))
    assert np.isnan(out[2, F["hip_height"]])
    assert np.isnan(out[1:4, F["hip_vy"]]).all()  # central differences touch both neighbours
    assert np.isfinite(out[0, F["hip_height"]]) and np.isfinite(out[4, F["hip_height"]])


def test_person_absent_rows_are_all_nan():
    kps, ts = sequence([300.0] * 7)
    present = np.array([True, False, True, True, True, True, True])
    out = frame_features(kps, ts, (H, W), person_present=present)
    assert np.isnan(out[1]).all()
    assert np.isfinite(out[2, [F["torso_angle_deg"], F["hip_height"], F["bbox_aspect"]]]).all()
    assert np.isnan(out[2, F["hip_vy"]])  # central difference touches the absent neighbour
    assert np.isfinite(out[5]).all()  # far enough from the gap for both derivatives


def test_all_zero_frame_from_extractor_is_nan_not_garbage():
    kps, ts = sequence([300.0] * 3)
    kps[1] = 0.0  # extract_keypoints.py writes zeros when no person is detected
    out = frame_features(kps, ts, (H, W))
    assert np.isnan(out[1, F["torso_angle_deg"]])
    assert np.isnan(out[1, F["hip_height"]])
    assert np.isnan(out[1, F["bbox_aspect"]])
    assert out[1, F["mean_conf"]] == 0.0


def test_min_conf_drops_low_confidence_joints():
    kps, ts = sequence([300.0] * 2)
    kps[:, [COCO17["l_hip"], COCO17["r_hip"]], 2] = 0.2
    assert np.isfinite(frame_features(kps, ts, (H, W))[:, F["hip_height"]]).all()
    assert np.isnan(frame_features(kps, ts, (H, W), min_conf=0.3)[:, F["hip_height"]]).all()


# ---- bbox -------------------------------------------------------------------------------------------


def test_detector_bbox_is_preferred_and_zero_box_falls_back():
    kps, ts = sequence([300.0] * 2)
    bbox = np.array([[0.0, 0.0, 200.0, 100.0], [0.0, 0.0, 0.0, 0.0]])
    out = frame_features(kps, ts, (H, W), bbox=bbox)
    assert np.isclose(out[0, F["bbox_aspect"]], 2.0)
    assert np.isclose(out[1, F["bbox_aspect"]], frame_features(kps, ts, (H, W))[1, F["bbox_aspect"]])


def test_bad_bbox_shape_raises():
    kps, ts = sequence([300.0] * 2)
    with pytest.raises(ValueError, match="bbox"):
        frame_features(kps, ts, (H, W), bbox=np.zeros((3, 4)))


# ---- window statistics -----------------------------------------------------------------------------


def test_window_stats_layout_and_nan_handling():
    feats = np.array(
        [
            [10.0, 0.5, 0.8, 0.0, np.nan, 0.9],
            [30.0, 0.7, 0.6, 1.0, np.nan, 0.8],
            [np.nan] * N_FEATURES,
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        s = window_stats(feats)
    assert s.shape == (4 * N_FEATURES,)
    angle = s[:4]
    assert np.allclose(angle, [10.0, 30.0, 20.0, 10.0])
    ay = s[4 * F["hip_ay"] : 4 * F["hip_ay"] + 4]
    assert np.isnan(ay).all()


def test_window_stats_empty_and_bad_shape():
    assert np.isnan(window_stats(np.zeros((0, N_FEATURES)))).all()
    with pytest.raises(ValueError):
        window_stats(np.zeros((3, N_FEATURES + 1)))


# ---- online extractor ----------------------------------------------------------------------------


def _frames(ys: list[float], present: bool = True) -> list[KeypointFrame]:
    return [
        KeypointFrame(ts=i / FPS, keypoints=skeleton((320.0, y)), person_present=present)
        for i, y in enumerate(ys)
    ]


def test_online_extractor_matches_offline_on_last_row():
    ys = [100.0 + 8.0 * i for i in range(30)]
    fx = FeatureExtractor((H, W), window_s=1.5)
    rows = [fx.push(f) for f in _frames(ys)]
    kps, ts = sequence(ys)
    # the online row is the trailing edge of its 1.5 s window (23 frames at 15 FPS), so it equals the last
    # row of `frame_features` run over exactly those frames (one-sided difference at the edge)
    for i in range(30):
        lo = max(0, i - 22)
        offline_window = frame_features(kps[lo : i + 1], ts[lo : i + 1], (H, W))
        assert np.allclose(rows[i], offline_window[-1], equal_nan=True), i


def test_online_window_is_trimmed_to_window_s():
    fx = FeatureExtractor((H, W), window_s=1.0)
    for f in _frames([300.0] * 40):
        fx.push(f)
    win = fx.window()
    assert win.shape == (16, N_FEATURES)  # 1.0 s at 15 FPS inclusive of both ends
    assert fx.timestamps()[-1] - fx.timestamps()[0] <= 1.0


def test_online_extractor_rejects_out_of_order_and_resets():
    fx = FeatureExtractor((H, W), window_s=1.0)
    frames = _frames([300.0, 300.0])
    fx.push(frames[1])
    with pytest.raises(ValueError, match="increasing"):
        fx.push(frames[0])
    fx.reset()
    assert fx.window().shape == (0, N_FEATURES)
    fx.push(frames[0])
    assert fx.window().shape == (1, N_FEATURES)


def test_online_extractor_absent_person_row_is_nan():
    fx = FeatureExtractor((H, W), window_s=1.0)
    fx.push(_frames([300.0])[0])
    row = fx.push(KeypointFrame(ts=0.5, keypoints=np.zeros((17, 3)), person_present=False))
    assert np.isnan(row).all()


def test_window_s_must_be_positive():
    with pytest.raises(ValueError):
        FeatureExtractor((H, W), window_s=0.0)
