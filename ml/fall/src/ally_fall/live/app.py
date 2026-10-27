"""The live demo's run loop and the two things a key press may save (F3c Task 5, D-69).

Nothing is stored unless the PO presses a save key and a folder was given (D-26): keypoints of
the last 10 s (no image), or the current frame.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from ally_fall.live.pipeline import FrameResult, LivePipeline

logger = logging.getLogger(__name__)


class ViewAction(Enum):
    NONE = "none"
    QUIT = "quit"
    RESET = "reset"
    SAVE_KEYPOINTS = "save_keypoints"
    SAVE_FRAME = "save_frame"


class Detector(Protocol):
    def detect(
        self, time_s: float, rgb: NDArray[np.uint8]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...


class Viewer(Protocol):
    def show(
        self, rgb: NDArray[np.uint8], result: FrameResult, pipeline: LivePipeline
    ) -> ViewAction: ...


@dataclass(frozen=True)
class RunSummary:
    frames: int
    ticks: int
    scored: int
    alerts: int
    seconds_of_video: float


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def save_keypoints(pipeline: LivePipeline, save_dir: Path | None) -> Path | None:
    """Write the keypoints of the last ticks (no image) to `save_dir`; None if no folder."""
    if save_dir is None:
        logger.warning("save keypoints: no --save-dir given, nothing saved")
        return None
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"keypoints-{_stamp()}.npz"
    np.savez(path, features=pipeline.recent_keypoints())
    logger.info("saved keypoints of the last ticks to %s", path)
    return path


def save_frame(rgb: NDArray[np.uint8], save_dir: Path | None) -> Path | None:
    """Write the current frame as a PNG to `save_dir`; None if no folder."""
    if save_dir is None:
        logger.warning("save frame: no --save-dir given, nothing saved")
        return None
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"frame-{_stamp()}.png"
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    logger.info("saved frame to %s", path)
    return path


def run(
    frames: Iterable[tuple[float, NDArray[np.uint8]]],
    detector: Detector,
    pipeline: LivePipeline,
    view: Viewer | None,
    save_dir: Path | None,
) -> RunSummary:
    """Pose, ticks, windows, scores and alerts for every frame, until the frames end or QUIT."""
    count = ticks = scored = 0
    last_time = first_time = 0.0
    for time_s, rgb in frames:
        if count == 0:
            first_time = time_s
        xy, vis = detector.detect(time_s, rgb)
        result = pipeline.process(time_s, xy, vis)
        count += 1
        last_time = time_s
        ticks += result.new_ticks
        scored += int(result.score is not None)
        if result.state is not None and result.state.fired:
            logger.info("ALERT at %.1f s (fall score %.2f)", time_s, result.score or 0.0)
        action = view.show(rgb, result, pipeline) if view is not None else ViewAction.NONE
        if action is ViewAction.QUIT:
            break
        if action is ViewAction.RESET:
            pipeline.reset()
        elif action is ViewAction.SAVE_KEYPOINTS:
            save_keypoints(pipeline, save_dir)
        elif action is ViewAction.SAVE_FRAME:
            save_frame(rgb, save_dir)
    return RunSummary(count, ticks, scored, pipeline.fired_total, last_time - first_time)
