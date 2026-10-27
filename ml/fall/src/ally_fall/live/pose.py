"""MediaPipe Pose Landmarker for the live demo: one person, VIDEO mode (D-13).

The same model file and settings as `keypoints.extract` (lite by default), one frame at a time.
`landmarks_to_arrays` and `next_timestamp_ms` are tested; `PoseDetector` needs the model file and
MediaPipe and is checked by hand (listed as an exclusion in the sprint review).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ally_fall.keypoints import POINTS


class Landmark(Protocol):
    x: float
    y: float
    visibility: float | None


class PoseResult(Protocol):
    pose_landmarks: Sequence[Sequence[Landmark]]


def landmarks_to_arrays(result: PoseResult) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """A landmarker result as xy [33, 2] and visibility [33]; no person gives zeros."""
    xy = np.zeros((POINTS, 2), dtype=np.float32)
    vis = np.zeros(POINTS, dtype=np.float32)
    if result.pose_landmarks:
        for i, point in enumerate(result.pose_landmarks[0][:POINTS]):
            xy[i] = (point.x, point.y)
            vis[i] = point.visibility if point.visibility is not None else 0.0
    return xy, vis


def next_timestamp_ms(time_s: float, previous_ms: int) -> int:
    """Whole milliseconds, strictly above the previous value (the landmarker requires it)."""
    return max(int(time_s * 1000.0), previous_ms + 1)


class PoseDetector:  # pragma: no cover - needs MediaPipe and the model file
    """`detect(time_s, rgb)` gives (xy [33, 2], visibility [33]) for one RGB frame."""

    def __init__(self, model_file: Path) -> None:
        if not model_file.is_file():
            raise FileNotFoundError(f"pose model file not found: {model_file}")
        import mediapipe as mp  # heavy; only the live demo needs it
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_file)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._previous_ms = -1

    def detect(
        self, time_s: float, rgb: NDArray[np.uint8]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
        )
        self._previous_ms = next_timestamp_ms(time_s, self._previous_ms)
        return landmarks_to_arrays(self._landmarker.detect_for_video(image, self._previous_ms))

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> PoseDetector:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
