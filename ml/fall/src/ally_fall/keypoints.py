"""Keypoint extraction with MediaPipe Pose Landmarker, storage, and verified deletion.

D-13: the same pose model as the phone (lite by default; full is also saved, PO
2026-10-19). D-54: a video (or frame folder) is deleted only once its keypoints are
saved and verified. Raw points are stored; points outside the image are marked
missing later, when ticks are built (ticks.mark_missing).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import zipfile
import zlib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from ally_fall.avi import dib_frames, is_uncompressed_avi

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
    """Frames of a video file with the decoder's own times (CAP_PROP_POS_MSEC).

    Uncompressed AVI (Le2i) is read by ally_fall.avi instead: OpenCV 5.0.0 corrupts memory
    converting those frames.
    """
    if is_uncompressed_avi(video):
        yield from dib_frames(video)
        return
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


# UP-Fall names each PNG by its capture time; the fraction is absent when it is exactly 0.
_FRAME_NAME = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2})(?:\.(\d{1,6}))?\.png$")


def _zip_images(archive: zipfile.ZipFile) -> list[str]:
    return [n for n in archive.namelist() if Path(n).suffix.lower() in IMAGE_SUFFIXES]


def _same_bytes(archive: zipfile.ZipFile, name: str, others: Iterable[str]) -> str | None:
    """The first of `others` whose content is byte for byte that of `name`, else None."""
    info = archive.getinfo(name)
    data: bytes | None = None
    for other in others:
        candidate = archive.getinfo(other)
        if (candidate.CRC, candidate.file_size) != (info.CRC, info.file_size):
            continue
        data = archive.read(name) if data is None else data
        if archive.read(other) == data:
            return other
    return None


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _cut_short_png(data: bytes) -> bool:
    """A PNG that starts as one but has no end marker (IEND): its writing was cut short."""
    return data.startswith(_PNG_SIGNATURE) and b"IEND" not in data


def _timed_images(archive: zipfile.ZipFile, clip: Path) -> list[tuple[datetime, str]]:
    """The time-named images in time order: the frames, counted and read alike.

    An image whose name has no time is skipped only if it is byte for byte a time-named
    frame of the same zip (Subject7Activity1Trial1Camera2.zip holds such a copy, checked
    2026-10-20, PO); any other one stops the clip. A frame whose PNG was cut short
    (Subject6Activity10Trial2Camera2.zip, 2026-10-20, PO) is skipped like a dropped frame;
    any other unreadable image still stops the clip when it is read.
    """
    timed: list[tuple[datetime, str]] = []
    misnamed: list[str] = []
    for name in _zip_images(archive):
        match = _FRAME_NAME.search(Path(name).name)
        if match is None:
            misnamed.append(name)
            continue
        if _cut_short_png(archive.read(name)):
            logger.warning("%s: skipped cut-short frame %s (no PNG end marker)", clip.name, name)
            continue
        stamp = datetime.strptime(match.group(1), "%Y-%m-%dT%H_%M_%S")
        fraction = match.group(2) or "0"
        timed.append((stamp.replace(microsecond=int(fraction.ljust(6, "0"))), name))
    for name in misnamed:
        original = _same_bytes(archive, name, (n for _, n in timed))
        if original is None:
            raise ValueError(f"{clip}: image name without a time: {name}")
        logger.warning("%s: skipped duplicate %s (same bytes as %s)", clip.name, name, original)
    timed.sort()
    for (earlier, _), (later, name) in zip(timed, timed[1:], strict=False):
        if later == earlier:
            raise ValueError(f"{clip}: two images have the same time ({name})")
    return timed


def zip_frame_count(clip: Path) -> int:
    """The number of frames in a zip of frames (the source count deletion is checked against)."""
    try:
        with zipfile.ZipFile(clip) as archive:
            return len(_timed_images(archive, clip))
    except (zipfile.BadZipFile, zlib.error, OSError) as error:
        raise ValueError(f"cannot open frame zip {clip}: {error}") from error


def zip_frames(clip: Path) -> Iterator[Frame]:
    """Frames of a zip of time-named PNGs (UP-Fall), in time order, times from the first frame.

    Both UP-Fall cameras and the sensor CSV share these timestamps (checked 2026-10-20 on
    Subject1Activity1Trial1), and OmniFall's labels end with the last frame.
    """
    try:
        with zipfile.ZipFile(clip) as archive:
            timed = _timed_images(archive, clip)
            if timed:
                logger.info("%s: first frame %s", clip.name, timed[0][0].isoformat())
            for stamp, name in timed:
                image = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"cannot read image {name} in {clip}")
                time_s = (stamp - timed[0][0]).total_seconds()
                yield time_s, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except (zipfile.BadZipFile, zlib.error, OSError) as error:
        raise ValueError(f"cannot read frame zip {clip}: {error}") from error


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
