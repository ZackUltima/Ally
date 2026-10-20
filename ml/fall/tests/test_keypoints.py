"""Tests for keypoint extraction, storage and verified deletion (D-13, D-54)."""

import os
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest

from ally_fall.keypoints import (
    Frame,
    KeypointSeq,
    extract,
    folder_frames,
    load_seq,
    save_seq,
    verify_and_delete,
    video_frames,
)

MODEL = Path(
    os.environ.get(
        "ALLY_POSE_MODEL",
        Path.home() / "ally-data" / "models" / "pose_landmarker_lite.task",
    )
)
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="pose model not downloaded")


def _write_video(path: Path, frames: int, fps: float = 25.0, size: int = 128) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (size, size))
    for _ in range(frames):
        writer.write(np.full((size, size, 3), 128, dtype=np.uint8))
    writer.release()
    return path


def _seq(frames: int) -> KeypointSeq:
    rng = np.random.default_rng(0)
    return KeypointSeq(
        dataset="GMDCSA24",
        video_id="Subject_1/ADL/01",
        subject=1,
        cam=1,
        pose_model="lite",
        frame_times_s=np.arange(frames) / 25.0,
        xy=rng.random((frames, 33, 2), dtype=np.float32),
        vis=rng.random((frames, 33), dtype=np.float32),
    )


def _extract(frames: Iterator[Frame]) -> KeypointSeq:
    return extract(
        frames, MODEL, dataset="test", video_id="v", subject=0, cam=1, pose_model="lite"
    )


def test_save_load_roundtrip(tmp_path: Path) -> None:
    seq = _seq(10)
    save_seq(seq, tmp_path / "a.npz")
    loaded = load_seq(tmp_path / "a.npz")
    assert (loaded.dataset, loaded.video_id, loaded.subject, loaded.cam, loaded.pose_model) == (
        "GMDCSA24",
        "Subject_1/ADL/01",
        1,
        1,
        "lite",
    )
    assert np.array_equal(loaded.frame_times_s, seq.frame_times_s)
    assert np.array_equal(loaded.xy, seq.xy) and np.array_equal(loaded.vis, seq.vis)
    assert not list(tmp_path.glob("*.tmp*"))


def test_video_frames_times(tmp_path: Path) -> None:
    frames = list(video_frames(_write_video(tmp_path / "v.avi", 10)))
    assert len(frames) == 10
    times = [t for t, _ in frames]
    assert times[0] == pytest.approx(0.0, abs=1e-3)
    assert all(b > a for a, b in zip(times, times[1:], strict=False))
    assert frames[0][1].shape == (128, 128, 3)


def test_folder_frames(tmp_path: Path) -> None:
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"{i:03d}.png"), np.zeros((8, 8, 3), dtype=np.uint8))
    frames = list(folder_frames(tmp_path, [0.0, 0.05, 0.1]))
    assert [t for t, _ in frames] == [0.0, 0.05, 0.1]
    with pytest.raises(ValueError):
        list(folder_frames(tmp_path, [0.0, 0.05]))


def test_rejects_empty_frames() -> None:
    with pytest.raises(ValueError, match="no frames"):
        _extract(iter([]))


@needs_model
def test_blank_video_all_missing(tmp_path: Path) -> None:
    seq = _extract(video_frames(_write_video(tmp_path / "v.avi", 50)))
    assert len(seq.frame_times_s) == 50
    assert seq.xy.shape == (50, 33, 2) and seq.vis.shape == (50, 33)
    assert not seq.vis.any()


@needs_model
def test_rejects_non_increasing_times() -> None:
    image = np.full((128, 128, 3), 128, dtype=np.uint8)
    with pytest.raises(ValueError, match="increase"):
        _extract(iter([(0.0, image), (0.04, image), (0.04, image)]))


def test_verify_and_delete_keeps_video_on_mismatch(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "v.avi", 5)
    save_seq(_seq(49), tmp_path / "a.npz")
    with pytest.raises(ValueError):
        verify_and_delete(video, tmp_path / "a.npz", expected_frames=50)
    assert video.exists()


def test_verify_and_delete_removes_video_on_match(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "v.avi", 5)
    save_seq(_seq(50), tmp_path / "a.npz")
    verify_and_delete(video, tmp_path / "a.npz", expected_frames=50)
    assert not video.exists()


def test_verify_and_delete_removes_frame_folder(tmp_path: Path) -> None:
    folder = tmp_path / "frames"
    folder.mkdir()
    (folder / "0.png").write_bytes(b"x")
    save_seq(_seq(1), tmp_path / "a.npz")
    verify_and_delete(folder, tmp_path / "a.npz", expected_frames=1)
    assert not folder.exists()


def test_verify_and_delete_keeps_video_when_saved_file_is_broken(tmp_path: Path) -> None:
    video = _write_video(tmp_path / "v.avi", 5)
    (tmp_path / "a.npz").write_bytes(b"not a zip")
    with pytest.raises(ValueError):
        verify_and_delete(video, tmp_path / "a.npz", expected_frames=5)
    assert video.exists()
