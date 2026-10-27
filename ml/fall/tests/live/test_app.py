"""Tests for the run loop, saving and the video-file source (F3c Task 5, D-69).

The camera, the pose model and the drawing are checked by hand; everything else here uses a
made-up detector and viewer."""

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from live_helpers import make_scorer

from ally_fall.keypoints import POINTS
from ally_fall.live.alert import AlertRule
from ally_fall.live.app import ViewAction, run, save_frame, save_keypoints
from ally_fall.live.pipeline import FrameResult, LivePipeline
from ally_fall.live.sources import file_frames


class _FakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(
        self, time_s: float, rgb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        rng = np.random.default_rng(self.calls)
        return (
            rng.uniform(0.1, 0.9, (POINTS, 2)).astype(np.float32),
            np.full(POINTS, 0.9, dtype=np.float32),
        )


class _ScriptedView:
    def __init__(self, actions: dict[int, ViewAction]) -> None:
        self._actions = actions
        self.shown = 0

    def show(self, rgb: np.ndarray, result: FrameResult, pipeline: LivePipeline) -> ViewAction:
        self.shown += 1
        return self._actions.get(self.shown, ViewAction.NONE)


def _frames(count: int, fps: float = 30.0) -> Iterator[tuple[float, np.ndarray]]:
    for i in range(count):
        yield i / fps, np.zeros((48, 64, 3), dtype=np.uint8)


def _pipeline(tmp_path: Path, threshold: float = 0.5) -> LivePipeline:
    return LivePipeline(make_scorer(tmp_path), AlertRule(threshold, 3, 1000))


def test_the_loop_runs_every_frame_and_summarises(tmp_path: Path) -> None:
    summary = run(_frames(150), _FakeDetector(), _pipeline(tmp_path), None, None)
    assert summary.frames == 150
    assert summary.ticks >= 49 and summary.scored >= 20
    assert summary.seconds_of_video > 4.9


def test_the_alert_count_is_reported(tmp_path: Path) -> None:
    summary = run(_frames(150), _FakeDetector(), _pipeline(tmp_path, 1e-9), None, None)
    assert summary.alerts == 1


def test_quit_stops_the_loop(tmp_path: Path) -> None:
    view = _ScriptedView({5: ViewAction.QUIT})
    summary = run(_frames(150), _FakeDetector(), _pipeline(tmp_path), view, None)
    assert summary.frames == 5 and view.shown == 5


def test_reset_clears_the_pipeline(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path, 1e-9)
    view = _ScriptedView({120: ViewAction.RESET})
    run(_frames(121), _FakeDetector(), pipeline, view, None)
    assert pipeline.fired_total == 0 and pipeline.last_score is None


def test_nothing_is_saved_unless_asked(tmp_path: Path) -> None:
    save_dir = tmp_path / "saved"
    run(_frames(60), _FakeDetector(), _pipeline(tmp_path), _ScriptedView({}), save_dir)
    assert not save_dir.exists()


def test_save_keypoints_writes_only_points_when_asked(tmp_path: Path) -> None:
    save_dir = tmp_path / "saved"
    view = _ScriptedView({50: ViewAction.SAVE_KEYPOINTS})
    run(_frames(60), _FakeDetector(), _pipeline(tmp_path), view, save_dir)
    files = list(save_dir.glob("keypoints-*.npz"))
    assert len(files) == 1
    with np.load(files[0]) as data:
        assert data["features"].shape[1:] == (POINTS, 3) and len(data["features"]) > 0
        assert set(data.files) == {"features"}  # no image in it


def test_save_without_a_folder_does_nothing(tmp_path: Path) -> None:
    pipeline = _pipeline(tmp_path)
    assert save_keypoints(pipeline, None) is None
    assert save_frame(np.zeros((4, 4, 3), np.uint8), None) is None


def test_save_frame_writes_one_image(tmp_path: Path) -> None:
    path = save_frame(np.full((8, 8, 3), 200, np.uint8), tmp_path / "f")
    assert path is not None and path.suffix == ".png" and path.is_file()


def test_file_frames_read_a_video_with_increasing_times(tmp_path: Path) -> None:
    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    for i in range(12):
        writer.write(np.full((48, 64, 3), i * 10, dtype=np.uint8))
    writer.release()
    frames = list(file_frames(video))
    assert len(frames) == 12
    times = [t for t, _ in frames]
    assert all(b > a for a, b in zip(times, times[1:], strict=False))
    assert frames[0][1].shape == (48, 64, 3) and frames[0][1].dtype == np.uint8
