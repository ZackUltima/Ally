"""Where the live demo's frames come from: the webcam or a video file (F3c Task 5, D-69).

A frame is (time in seconds, RGB image), as everywhere in `keypoints`. Video files use the F1
reader, so a recorded clip goes through exactly the path the training data did.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from pathlib import Path

import cv2

from ally_fall.keypoints import Frame, video_frames


def file_frames(video: Path) -> Iterator[Frame]:
    """Frames of a video file with the decoder's own times."""
    return video_frames(video)


def webcam_frames(index: int = 0) -> Iterator[Frame]:  # pragma: no cover - needs a camera
    """Frames from a webcam, timed by the laptop clock from the first frame."""
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    capture = cv2.VideoCapture(index, backend)
    if not capture.isOpened():
        raise ValueError(f"cannot open camera {index}")
    start = time.perf_counter()
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                continue
            yield time.perf_counter() - start, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()
