"""Tests for the per-dataset extraction runner (D-54: delete only after verifying)."""

import zipfile
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
import pytest

from ally_fall.build import keypoint_path
from ally_fall.datasets import VideoItem
from ally_fall.extract_dataset import run, source_frame_count, source_frames
from ally_fall.keypoints import Frame, KeypointSeq, load_seq


def _fake_extractor(frames_per_video: int, fail_for: str | None = None):  # noqa: ANN202
    def extractor(
        frames: Iterable[Frame], model_file: Path, *, dataset: str, video_id: str,
        subject: int, cam: int, pose_model: str,
    ) -> KeypointSeq:
        if pose_model == fail_for:
            raise ValueError("model failed")
        return KeypointSeq(
            dataset=dataset, video_id=video_id, subject=subject, cam=cam, pose_model=pose_model,
            frame_times_s=np.arange(frames_per_video) / 25.0,
            xy=np.zeros((frames_per_video, 33, 2), dtype=np.float32),
            vis=np.zeros((frames_per_video, 33), dtype=np.float32),
        )
    return extractor


def _png() -> bytes:
    return cv2.imencode(".png", np.zeros((4, 4, 3), dtype=np.uint8))[1].tobytes()


def _no_frames(_: VideoItem) -> Iterable[Frame]:
    return []


@pytest.fixture()
def item(tmp_path: Path) -> VideoItem:
    source = tmp_path / "raw" / "01.mp4"
    source.parent.mkdir()
    source.write_bytes(b"video")
    return VideoItem(source=source, video_id="Subject_1/ADL/01", subject=1, cam=1)


MODELS = {"lite": Path("lite.task"), "full": Path("full.task")}


def test_saves_both_models_then_deletes(tmp_path: Path, item: VideoItem) -> None:
    report = run([item], MODELS, tmp_path, "GMDCSA24", delete=True,
                 extractor=_fake_extractor(10), frames_of=_no_frames)
    for model in MODELS:
        saved = keypoint_path(tmp_path, model, "GMDCSA24", item.video_id, 1)
        assert load_seq(saved).pose_model == model
    assert not item.source.exists()
    assert report.done == [item.video_id] and report.failed == []


def test_keeps_source_when_delete_is_off(tmp_path: Path, item: VideoItem) -> None:
    run([item], MODELS, tmp_path, "GMDCSA24", delete=False,
        extractor=_fake_extractor(10), frames_of=_no_frames)
    assert item.source.exists()


def test_keeps_source_when_one_model_fails(tmp_path: Path, item: VideoItem) -> None:
    report = run([item], MODELS, tmp_path, "GMDCSA24", delete=True,
                 extractor=_fake_extractor(10, fail_for="full"), frames_of=_no_frames)
    assert item.source.exists()
    assert report.failed == [(item.video_id, "full: model failed")]


def test_resumes_without_redoing_saved_videos(tmp_path: Path, item: VideoItem) -> None:
    run([item], MODELS, tmp_path, "GMDCSA24", delete=False,
        extractor=_fake_extractor(10), frames_of=_no_frames)
    report = run([item], MODELS, tmp_path, "GMDCSA24", delete=False,
                 extractor=_fake_extractor(10, fail_for="lite"), frames_of=_no_frames)
    assert report.already_done == [item.video_id] and report.failed == []


def test_cli_with_no_videos_returns_0(tmp_path: Path) -> None:
    from ally_fall.extract_dataset import main

    (tmp_path / "raw").mkdir()
    assert main([
        "--data-root", str(tmp_path), "--dataset", "GMDCSA24", "--raw", str(tmp_path / "raw"),
        "--models-dir", str(tmp_path),
    ]) == 0


def test_cli_le2i_reads_subjects_from_omnifall(tmp_path: Path) -> None:
    from ally_fall.extract_dataset import main

    labels = tmp_path / "omnifall" / "labels"
    labels.mkdir(parents=True)
    (labels / "le2i.csv").write_text("path,label,start,end,subject,cam,dataset\n")
    (tmp_path / "raw").mkdir()
    assert main([
        "--data-root", str(tmp_path), "--dataset", "le2i", "--raw", str(tmp_path / "raw"),
        "--models-dir", str(tmp_path),
    ]) == 0


def test_cli_missing_model_returns_1(tmp_path: Path) -> None:
    from ally_fall.extract_dataset import main

    raw = tmp_path / "raw" / "Subject 1" / "ADL"
    raw.mkdir(parents=True)
    (raw / "01.mp4").write_bytes(b"")
    assert main([
        "--data-root", str(tmp_path), "--dataset", "GMDCSA24", "--raw", str(tmp_path / "raw"),
        "--models-dir", str(tmp_path / "nowhere"),
    ]) == 1


def test_keeps_source_when_fewer_frames_than_the_source_holds(
    tmp_path: Path, item: VideoItem
) -> None:
    report = run([item], MODELS, tmp_path, "up_fall", delete=True,
                 extractor=_fake_extractor(10), frames_of=_no_frames,
                 frame_count_of=lambda _: 12)
    assert item.source.exists()
    assert report.failed == [(item.video_id, "lite: 10 frames extracted, 12 in the source")]


def test_deletes_when_frames_match_the_source(tmp_path: Path, item: VideoItem) -> None:
    report = run([item], MODELS, tmp_path, "up_fall", delete=True,
                 extractor=_fake_extractor(10), frames_of=_no_frames,
                 frame_count_of=lambda _: 10)
    assert not item.source.exists() and report.done == [item.video_id]


def test_zip_sources_are_read_and_counted_from_the_zip(tmp_path: Path) -> None:
    clip = tmp_path / "Subject1Activity1Trial1Camera1.zip"
    with zipfile.ZipFile(clip, "w") as archive:
        archive.writestr("2018-07-04T12_04_17.738369.png", _png())
        archive.writestr("2018-07-04T12_04_17.790509.png", _png())
    zipped = VideoItem(source=clip, video_id="x", subject=1, cam=1)
    assert source_frame_count(zipped) == 2
    assert [round(t, 6) for t, _ in source_frames(zipped)] == [0.0, 0.05214]
    assert source_frame_count(VideoItem(source=tmp_path / "v.mp4", video_id="y",
                                        subject=1, cam=1)) is None


def test_cli_accepts_up_fall(tmp_path: Path) -> None:
    from ally_fall.extract_dataset import main

    labels = tmp_path / "omnifall" / "labels"
    labels.mkdir(parents=True)
    (labels / "up_fall.csv").write_text("path,label,start,end,subject,cam,dataset\n")
    (tmp_path / "raw").mkdir()
    assert main([
        "--data-root", str(tmp_path), "--dataset", "up_fall", "--raw", str(tmp_path / "raw"),
        "--models-dir", str(tmp_path),
    ]) == 0
