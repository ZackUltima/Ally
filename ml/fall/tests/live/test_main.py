"""Tests for the command line of the live demo (F3c Task 5, D-69). The window and the camera
are checked by hand; here a made-up detector and a short video file stand in."""

import logging
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from live_helpers import make_scorer

from ally_fall.keypoints import POINTS
from ally_fall.live.__main__ import build_parser, limit_seconds, main


class _Detector:
    def detect(
        self, time_s: float, rgb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.full((POINTS, 2), 0.5, dtype=np.float32),
            np.full(POINTS, 0.9, dtype=np.float32),
        )


def _video(path: Path, frames: int = 80, fps: float = 10.0) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    for i in range(frames):
        writer.write(np.full((48, 64, 3), i, dtype=np.uint8))
    writer.release()
    return path


def test_headless_run_on_a_video_exits_cleanly_and_logs_a_summary(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    make_scorer(tmp_path)
    video = _video(tmp_path / "clip.avi")
    argv = [
        "--source", str(video), "--headless",
        "--config", str(tmp_path / "run.toml"), "--weights", str(tmp_path / "best.pt"),
        "--omnifall-dir", str(tmp_path / "omnifall"), "--pose-model", str(tmp_path / "pose.task"),
    ]
    with caplog.at_level(logging.INFO):
        code = main(argv, detector=_Detector())
    assert code == 0
    assert any("80 frames" in r.getMessage() for r in caplog.records)


def test_defaults_match_the_validation_table_and_the_plan() -> None:
    args = build_parser().parse_args([])
    assert args.threshold == 0.80 and args.required == 3 and args.cool_down == 300
    assert args.source == "0" and args.run == "tcn-omnifall16-b0.999-s0"
    assert args.save_dir is None  # nothing is stored unless a folder is given


def test_limit_seconds_stops_after_the_time() -> None:
    def frames() -> Iterator[tuple[float, np.ndarray]]:
        for i in range(1000):
            yield i / 10.0, np.zeros((2, 2, 3), np.uint8)

    assert len(list(limit_seconds(frames(), 5.0))) == 51  # 0.0 .. 5.0 s
    assert len(list(limit_seconds(frames(), None))) == 1000


def test_a_missing_checkpoint_gives_a_clear_error(tmp_path: Path) -> None:
    make_scorer(tmp_path)
    argv = [
        "--source", str(_video(tmp_path / "c.avi")), "--headless",
        "--config", str(tmp_path / "run.toml"), "--weights", str(tmp_path / "missing.pt"),
        "--omnifall-dir", str(tmp_path / "omnifall"), "--pose-model", str(tmp_path / "pose.task"),
    ]
    with pytest.raises(FileNotFoundError):
        main(argv, detector=_Detector())
