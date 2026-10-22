"""Tests for keypoint extraction, storage and verified deletion (D-13, D-54)."""

import os
import zipfile
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
    zip_frame_count,
    zip_frames,
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


# UP-Fall ships one zip of PNG frames per clip, each named by its capture time
# (real names, Subject1Activity1Trial1Camera1.zip, 2026-10-20).
def _png(value: int) -> bytes:
    ok, encoded = cv2.imencode(".png", np.full((8, 8, 3), value, dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def _frame_zip(path: Path, names: list[str]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for i, name in enumerate(names):
            archive.writestr(name, _png(10 * i))
    return path


def test_zip_frames_times_from_names(tmp_path: Path) -> None:
    clip = _frame_zip(tmp_path / "c.zip", [
        "2018-07-04T12_04_17.790509.png",
        "2018-07-04T12_04_18.png",  # no fraction when the microseconds are 0
        "2018-07-04T12_04_17.738369.png",
    ])
    frames = list(zip_frames(clip))
    assert [round(t, 6) for t, _ in frames] == [0.0, 0.05214, 0.261631]
    assert frames[0][1].shape == (8, 8, 3) and frames[0][1][0, 0, 0] == 20
    assert zip_frame_count(clip) == 3


def test_zip_frame_count_ignores_other_members(tmp_path: Path) -> None:
    clip = _frame_zip(tmp_path / "c.zip", ["2018-07-04T12_04_17.738369.png"])
    with zipfile.ZipFile(clip, "a") as archive:
        archive.writestr("folder/", b"")
    assert zip_frame_count(clip) == 1
    assert len(list(zip_frames(clip))) == 1


def test_zip_frames_rejects_duplicate_times(tmp_path: Path) -> None:
    clip = _frame_zip(tmp_path / "c.zip", ["a/2018-07-04T12_04_18.png", "2018-07-04T12_04_18.png"])
    with pytest.raises(ValueError, match="same time"):
        list(zip_frames(clip))


def test_zip_frames_rejects_unknown_image_name(tmp_path: Path) -> None:
    clip = _frame_zip(tmp_path / "c.zip", ["frame_001.png"])
    with pytest.raises(ValueError, match="frame_001"):
        list(zip_frames(clip))


def test_zip_frames_skips_misnamed_identical_copy(tmp_path: Path) -> None:
    # Real case, Subject7Activity1Trial1Camera2.zip (2026-10-20): an extra image named
    # "2018-07-06T12_42_47.2340654 - copi.png", byte for byte the frame ...47.234065.png.
    clip = _frame_zip(tmp_path / "c.zip", [
        "2018-07-06T12_42_47.170896.png",
        "2018-07-06T12_42_47.234065.png",
    ])
    with zipfile.ZipFile(clip, "a") as archive:
        archive.writestr("2018-07-06T12_42_47.2340654 - copi.png", _png(10))
    frames = list(zip_frames(clip))
    assert zip_frame_count(clip) == len(frames) == 2
    assert [round(t, 6) for t, _ in frames] == [0.0, 0.063169]


def test_zip_frames_rejects_misnamed_image_with_other_content(tmp_path: Path) -> None:
    clip = _frame_zip(tmp_path / "c.zip", ["2018-07-06T12_42_47.234065.png"])
    with zipfile.ZipFile(clip, "a") as archive:
        archive.writestr("2018-07-06T12_42_47.2340654 - copi.png", _png(99))
    with pytest.raises(ValueError, match="copi"):
        list(zip_frames(clip))
    with pytest.raises(ValueError, match="copi"):
        zip_frame_count(clip)


def _cut_short(png: bytes) -> bytes:
    # Real case, Subject6Activity10Trial2Camera2.zip (2026-10-20): a frame of normal size
    # that ends in zeros with no PNG end marker (IEND).
    return png[: len(png) // 2] + bytes(len(png) - len(png) // 2)


def test_zip_frames_skips_cut_short_png(tmp_path: Path) -> None:
    clip = tmp_path / "c.zip"
    with zipfile.ZipFile(clip, "w") as archive:
        archive.writestr("2018-07-06T12_03_04.429377.png", _png(10))
        archive.writestr("2018-07-06T12_03_04.483526.png", _cut_short(_png(20)))
        archive.writestr("2018-07-06T12_03_04.566753.png", _png(30))
    frames = list(zip_frames(clip))
    assert zip_frame_count(clip) == len(frames) == 2
    assert [round(t, 6) for t, _ in frames] == [0.0, 0.137376]
    assert frames[1][1][0, 0, 0] == 30


def test_zip_frames_rejects_damaged_png_with_end_marker(tmp_path: Path) -> None:
    png = bytearray(_png(20))
    png[16:24] = bytes(8)  # damage the header chunk; the end marker stays
    clip = tmp_path / "c.zip"
    with zipfile.ZipFile(clip, "w") as archive:
        archive.writestr("2018-07-06T12_03_04.429377.png", _png(10))
        archive.writestr("2018-07-06T12_03_04.483526.png", bytes(png))
    with pytest.raises(ValueError, match="cannot read image"):
        list(zip_frames(clip))


def test_zip_frames_corrupt_download_is_value_error(tmp_path: Path) -> None:
    page = tmp_path / "c.zip"
    page.write_bytes(b"<html>Quota exceeded</html>")
    with pytest.raises(ValueError, match=r"c\.zip"):
        list(zip_frames(page))
    with pytest.raises(ValueError, match=r"c\.zip"):
        zip_frame_count(page)


def test_zip_frames_unreadable_image_is_value_error(tmp_path: Path) -> None:
    clip = tmp_path / "c.zip"
    with zipfile.ZipFile(clip, "w") as archive:
        archive.writestr("2018-07-04T12_04_18.png", b"not an image")
    with pytest.raises(ValueError, match="cannot read image"):
        list(zip_frames(clip))
