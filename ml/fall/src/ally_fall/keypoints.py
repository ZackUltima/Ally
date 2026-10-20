"""Keypoint extraction with MediaPipe Pose Landmarker, storage, and verified deletion.

D-13: the same pose model as the phone (lite by default; full is also saved, PO
2026-10-19). D-54: a video (or frame folder) is deleted only once its keypoints are
saved and verified. Raw points are stored; points outside the image are marked
missing later, when ticks are built (ticks.mark_missing).
"""

from __future__ import annotations

import logging
import os
import shutil
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

POINTS: int = 33
IMAGE_SUFFIXES: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".bmp"})

Frame = tuple[float, NDArray[np.uint8]]  # (time in seconds, RGB image)


@dataclass(frozen=True)
class KeypointSeq:
    """One video's keypoints: [T] times, [T, 33, 2] normalised x and y, [T, 33] visibility."""

    dataset: str
    video_id: str
    subject: int
    cam: int
    pose_model: str
    frame_times_s: NDArray[np.float64]
    xy: NDArray[np.float32]
    vis: NDArray[np.float32]


def save_seq(seq: KeypointSeq, path: Path) -> None:
    """Write to a temporary file, then rename, so a crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as file:
        np.savez(
            file,
            dataset=np.array(seq.dataset),
            video_id=np.array(seq.video_id),
            subject=np.array(seq.subject, dtype=np.int64),
            cam=np.array(seq.cam, dtype=np.int64),
            pose_model=np.array(seq.pose_model),
            frame_times_s=seq.frame_times_s.astype(np.float64),
            xy=seq.xy.astype(np.float32),
            vis=seq.vis.astype(np.float32),
        )
    os.replace(temporary, path)


def load_seq(path: Path) -> KeypointSeq:
    with np.load(path, allow_pickle=False) as data:
        return KeypointSeq(
            dataset=str(data["dataset"]),
            video_id=str(data["video_id"]),
            subject=int(data["subject"]),
            cam=int(data["cam"]),
            pose_model=str(data["pose_model"]),
            frame_times_s=data["frame_times_s"].astype(np.float64),
            xy=data["xy"].astype(np.float32),
            vis=data["vis"].astype(np.float32),
        )


def video_frames(video: Path) -> Iterator[Frame]:
    """Frames of a video file with the decoder's own times (CAP_PROP_POS_MSEC)."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cannot open video {video}")
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            # With the FFmpeg backend, POS_MSEC after read() is the time of the frame just read.
            time_s = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            yield time_s, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def folder_frames(folder: Path, times_s: Sequence[float]) -> Iterator[Frame]:
    """Frames from a folder of images (sorted by name) with their times given separately."""
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if len(files) != len(times_s):
        raise ValueError(f"{folder}: {len(files)} images but {len(times_s)} times")
    for time_s, file in zip(times_s, files, strict=True):
        image = cv2.imread(str(file), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot read image {file}")
        yield float(time_s), cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def extract(
    frames: Iterable[Frame],
    model_file: Path,
    *,
    dataset: str,
    video_id: str,
    subject: int,
    cam: int,
    pose_model: str,
) -> KeypointSeq:
    """Run the Pose Landmarker (VIDEO mode, one pose) on every frame.

    A frame with no person gives 33 points with visibility 0. Raises ValueError if there
    are no frames or the frame times do not increase.
    """
    iterator = iter(frames)
    first = next(iterator, None)
    if first is None:
        raise ValueError(f"{dataset} {video_id}: no frames")

    import mediapipe as mp  # imported here: heavy, and only extraction needs it
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_file)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    times: list[float] = []
    xy: list[NDArray[np.float32]] = []
    vis: list[NDArray[np.float32]] = []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for time_s, rgb in chain([first], iterator):
            if times and time_s <= times[-1]:
                raise ValueError(
                    f"{dataset} {video_id}: frame times do not increase at {time_s:.4f} s"
                )
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            result = landmarker.detect_for_video(image, int(round(time_s * 1000.0)))
            frame_xy = np.zeros((POINTS, 2), dtype=np.float32)
            frame_vis = np.zeros(POINTS, dtype=np.float32)
            if result.pose_landmarks:
                for i, point in enumerate(result.pose_landmarks[0][:POINTS]):
                    frame_xy[i] = (point.x, point.y)
                    frame_vis[i] = point.visibility if point.visibility is not None else 0.0
            times.append(time_s)
            xy.append(frame_xy)
            vis.append(frame_vis)
    return KeypointSeq(
        dataset=dataset,
        video_id=video_id,
        subject=subject,
        cam=cam,
        pose_model=pose_model,
        frame_times_s=np.asarray(times, dtype=np.float64),
        xy=np.stack(xy),
        vis=np.stack(vis),
    )


def verify_saved(saved: Path, expected_frames: int) -> None:
    """Raise ValueError unless the saved keypoints load and hold the expected frame count."""
    try:
        seq = load_seq(saved)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        raise ValueError(f"{saved}: saved keypoints unreadable") from error
    if len(seq.frame_times_s) != expected_frames:
        raise ValueError(
            f"{saved}: {len(seq.frame_times_s)} frames saved, {expected_frames} expected"
        )


def verify_and_delete(source: Path, saved: Path, expected_frames: int) -> None:
    """Delete the video file or frame folder only if the saved keypoints load and match."""
    try:
        verify_saved(saved, expected_frames)
    except ValueError as error:
        raise ValueError(f"{error}; keeping {source}") from error
    if source.is_dir():
        shutil.rmtree(source)
    else:
        source.unlink()
    logger.info("deleted %s after verifying %s", source, saved)
