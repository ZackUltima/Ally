"""Per-frame kinematic features over a keypoint stream (methods.md §6.1 step 2) and their window statistics.

Pure numpy, no I/O, no clock: time comes from the frame timestamps. The feature set is fixed by the plan:
torso angle to vertical (shoulder-centre → hip-centre), bounding-box aspect ratio, normalised hip height,
vertical velocity and acceleration of the hip centre (finite differences over `ts`), mean keypoint confidence.

Conventions (image coordinates: x right, y down, in pixels of the capture frame)
    hip_height   1 - hip_y / H          1.0 = top of frame, 0.0 = bottom
    hip_vy       d(hip_y / H) / dt      frame-heights per second, **positive = moving down**
    hip_ay       d(hip_vy) / dt
    torso_angle  degrees from vertical  0° upright, 90° horizontal, > 90° head below hips
    bbox_aspect  width / height         > 1 when the body is wider than tall (lying)
    missing data → NaN                  a keypoint with conf <= `min_conf`, a frame without a person, or a
                                        derivative next to such a frame; window stats are NaN-aware

Layout is COCO-17 (YOLO-pose). MediaPipe's 33-point layout gets its own index mapping if it wins in
Sprint 1 — only `COCO17` and `_JOINTS` know which row is which joint.
"""

from __future__ import annotations

import warnings
from collections import deque
from typing import Final

import numpy as np

from ally.contracts import KeypointFrame

# --------------------------------------------------------------------------------------------------
# Keypoint layout
# --------------------------------------------------------------------------------------------------

COCO17: Final[dict[str, int]] = {
    "nose": 0,
    "l_eye": 1,
    "r_eye": 2,
    "l_ear": 3,
    "r_ear": 4,
    "l_shoulder": 5,
    "r_shoulder": 6,
    "l_elbow": 7,
    "r_elbow": 8,
    "l_wrist": 9,
    "r_wrist": 10,
    "l_hip": 11,
    "r_hip": 12,
    "l_knee": 13,
    "r_knee": 14,
    "l_ankle": 15,
    "r_ankle": 16,
}
N_KPTS_COCO: Final[int] = 17
_JOINTS: Final[dict[str, tuple[int, int]]] = {
    "shoulders": (COCO17["l_shoulder"], COCO17["r_shoulder"]),
    "hips": (COCO17["l_hip"], COCO17["r_hip"]),
}

# --------------------------------------------------------------------------------------------------
# Feature names (column order of every feature array produced here)
# --------------------------------------------------------------------------------------------------

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "torso_angle_deg",
    "bbox_aspect",
    "hip_height",
    "hip_vy",
    "hip_ay",
    "mean_conf",
)
N_FEATURES: Final[int] = len(FEATURE_NAMES)
F: Final[dict[str, int]] = {name: i for i, name in enumerate(FEATURE_NAMES)}

WINDOW_STATS: Final[tuple[str, ...]] = ("min", "max", "mean", "std")
WINDOW_FEATURE_NAMES: Final[tuple[str, ...]] = tuple(
    f"{name}_{stat}" for name in FEATURE_NAMES for stat in WINDOW_STATS
)


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def _validate(keypoints: np.ndarray, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    kp = np.asarray(keypoints, dtype=np.float64)
    t = np.asarray(ts, dtype=np.float64)
    if kp.ndim != 3 or kp.shape[2] != 3:
        raise ValueError(f"keypoints must have shape [T, N, 3], got {kp.shape}")
    if t.ndim != 1 or t.shape[0] != kp.shape[0]:
        raise ValueError(f"ts must have shape [T] = [{kp.shape[0]}], got {t.shape}")
    if t.shape[0] > 1 and not np.all(np.diff(t) > 0):
        raise ValueError("ts must be strictly increasing")
    return kp, t


def _joint_centre(kp: np.ndarray, valid: np.ndarray, idx: tuple[int, int]) -> np.ndarray:
    """Mean (x, y) of the two joints that are valid; one joint alone is used if the other is missing;
    NaN when both are missing. kp: [T, N, 3], valid: [T, N] → [T, 2]."""
    pts = kp[:, idx, :2]  # [T, 2, 2]
    ok = valid[:, idx]  # [T, 2]
    n = ok.sum(axis=1)  # [T]
    summed = np.where(ok[..., None], pts, 0.0).sum(axis=1)  # [T, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        centre = summed / n[:, None]
    centre[n == 0] = np.nan
    return centre


def _bbox_from_keypoints(kp: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """[T, 4] x1, y1, x2, y2 spanned by the valid keypoints; NaN when fewer than two are valid."""
    x = np.where(valid, kp[..., 0], np.nan)
    y = np.where(valid, kp[..., 1], np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        box = np.stack([np.nanmin(x, 1), np.nanmin(y, 1), np.nanmax(x, 1), np.nanmax(y, 1)], axis=1)
    box[valid.sum(axis=1) < 2] = np.nan
    return box


def _gradient(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """d y / d t by central finite differences on the (possibly irregular) timestamps; NaN with T < 2."""
    if y.shape[0] < 2:
        return np.full_like(y, np.nan)
    return np.gradient(y, t)


# --------------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------------


def frame_features(
    keypoints: np.ndarray,
    ts: np.ndarray,
    frame_hw: tuple[int, int],
    *,
    bbox: np.ndarray | None = None,
    person_present: np.ndarray | None = None,
    min_conf: float = 0.0,
) -> np.ndarray:
    """Per-frame features for a whole sequence.

    keypoints       [T, 17, 3] (x_px, y_px, conf) in COCO-17 order, as written by scripts/extract_keypoints.py
    ts              [T] seconds, strictly increasing
    frame_hw        (height, width) of the capture frame in pixels — normalises hip height
    bbox            optional [T, 4] detector boxes (x1, y1, x2, y2); zero-area boxes fall back to the
                    keypoint extent
    person_present  optional [T] bool; False → the whole row is NaN
    min_conf        a keypoint with conf <= min_conf is missing (0.0 = only undetected joints are dropped;
                    a tuned value is a Sprint 2 decision)
    returns         [T, N_FEATURES] float64, columns in FEATURE_NAMES order
    """
    kp, t = _validate(keypoints, ts)
    n_frames = kp.shape[0]
    height = float(frame_hw[0])
    if height <= 0:
        raise ValueError("frame height must be positive")

    valid = kp[..., 2] > min_conf  # [T, N]
    present = None
    if person_present is not None:
        present = np.asarray(person_present, dtype=bool)
        if present.shape != (n_frames,):
            raise ValueError(f"person_present must have shape [{n_frames}], got {present.shape}")
        valid &= present[:, None]

    shoulders = _joint_centre(kp, valid, _JOINTS["shoulders"])  # [T, 2]
    hips = _joint_centre(kp, valid, _JOINTS["hips"])  # [T, 2]

    # torso angle: 0° when the hip centre is straight below the shoulder centre
    d = hips - shoulders
    torso_angle = np.degrees(np.arctan2(np.abs(d[:, 0]), d[:, 1]))

    # bbox aspect: detector box when it has area, else the keypoint extent
    derived = _bbox_from_keypoints(kp, valid)
    if bbox is not None:
        given = np.asarray(bbox, dtype=np.float64)
        if given.shape != (n_frames, 4):
            raise ValueError(f"bbox must have shape [{n_frames}, 4], got {given.shape}")
        has_area = (given[:, 2] > given[:, 0]) & (given[:, 3] > given[:, 1])
        box = np.where(has_area[:, None], given, derived)
    else:
        box = derived
    with np.errstate(invalid="ignore", divide="ignore"):
        bbox_aspect = (box[:, 2] - box[:, 0]) / (box[:, 3] - box[:, 1])

    # hip height and its derivatives (normalised by frame height)
    hip_y_norm = hips[:, 1] / height
    hip_height = 1.0 - hip_y_norm
    hip_vy = _gradient(hip_y_norm, t)
    hip_ay = _gradient(hip_vy, t)

    mean_conf = kp[..., 2].mean(axis=1)

    out = np.stack([torso_angle, bbox_aspect, hip_height, hip_vy, hip_ay, mean_conf], axis=1)
    if present is not None:
        out[~present] = np.nan
    return out


def window_stats(features: np.ndarray) -> np.ndarray:
    """min / max / mean / std of every feature over one window (NaN-aware; an all-NaN column stays NaN).

    features  [T, N_FEATURES]  →  [N_FEATURES * 4] in WINDOW_FEATURE_NAMES order
    """
    f = np.asarray(features, dtype=np.float64)
    if f.ndim != 2 or f.shape[1] != N_FEATURES:
        raise ValueError(f"features must have shape [T, {N_FEATURES}], got {f.shape}")
    if f.shape[0] == 0:
        return np.full(N_FEATURES * len(WINDOW_STATS), np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stats = np.stack([np.nanmin(f, 0), np.nanmax(f, 0), np.nanmean(f, 0), np.nanstd(f, 0)], axis=1)
    return stats.reshape(-1)  # feature-major: f0_min, f0_max, f0_mean, f0_std, f1_min, ...


class FeatureExtractor:
    """Online counterpart of `frame_features`: push one KeypointFrame at a time, read the newest row.

    Keeps the last `window_s` seconds of frames so derivatives and `window()` are available live; the
    per-frame numbers are identical to running `frame_features` over the same frames.
    """

    def __init__(self, frame_hw: tuple[int, int], window_s: float, *, min_conf: float = 0.0) -> None:
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        self._frame_hw = frame_hw
        self._window_s = float(window_s)
        self._min_conf = min_conf
        self._frames: deque[KeypointFrame] = deque()
        self._features: np.ndarray = np.zeros((0, N_FEATURES))

    @property
    def window_s(self) -> float:
        return self._window_s

    def reset(self) -> None:
        self._frames.clear()
        self._features = np.zeros((0, N_FEATURES))

    def push(self, frame: KeypointFrame) -> np.ndarray:
        """Add a frame; returns its feature row [N_FEATURES]."""
        if self._frames and frame.ts <= self._frames[-1].ts:
            raise ValueError("frames must arrive with strictly increasing ts")
        self._frames.append(frame)
        while self._frames and frame.ts - self._frames[0].ts > self._window_s:
            self._frames.popleft()
        self._features = frame_features(
            np.stack([f.keypoints for f in self._frames]),
            np.array([f.ts for f in self._frames]),
            self._frame_hw,
            bbox=np.array([f.bbox if f.bbox is not None else (0.0, 0.0, 0.0, 0.0) for f in self._frames]),
            person_present=np.array([f.person_present for f in self._frames]),
            min_conf=self._min_conf,
        )
        return self._features[-1]

    def window(self) -> np.ndarray:
        """Features of the buffered frames, oldest first: [T, N_FEATURES] with T covering ≤ window_s."""
        return self._features.copy()

    def timestamps(self) -> np.ndarray:
        return np.array([f.ts for f in self._frames])


__all__ = [
    "COCO17",
    "F",
    "FEATURE_NAMES",
    "N_FEATURES",
    "N_KPTS_COCO",
    "WINDOW_FEATURE_NAMES",
    "WINDOW_STATS",
    "FeatureExtractor",
    "frame_features",
    "window_stats",
]
