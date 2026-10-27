"""Command line of the laptop live demo (sprint F3c, D-69).

    python -m ally_fall.live --source 0                      # laptop webcam, window with keys
    python -m ally_fall.live --source clip.mp4 --headless    # a video file, no window

Keys: q quit, space pause, r reset, + / - threshold, [ / ] ticks in a row, m mirror,
s save keypoints of the last 10 s, f save the current frame (both need --save-dir).
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ally_fall.live.alert import AlertRule
from ally_fall.live.app import Detector, Viewer, run
from ally_fall.live.pipeline import LivePipeline
from ally_fall.live.scorer import Scorer

logger = logging.getLogger(__name__)

DEFAULT_RUN = "tcn-omnifall16-b0.999-s0"


def build_parser() -> argparse.ArgumentParser:
    models = Path.home() / "ally-tools" / "models"
    parser = argparse.ArgumentParser(prog="python -m ally_fall.live", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0", help="camera number, or a video file path")
    parser.add_argument("--run", default=DEFAULT_RUN, help="folder name under --models-dir")
    parser.add_argument("--models-dir", type=Path, default=models)
    parser.add_argument("--config", type=Path, help="default: <models-dir>/<run>/run.toml")
    parser.add_argument("--weights", type=Path, help="default: <models-dir>/<run>/best.pt")
    parser.add_argument("--omnifall-dir", type=Path, help="default: <models-dir>/omnifall")
    parser.add_argument("--pose-model", type=Path,
                        help="default: <models-dir>/pose_landmarker_lite.task")
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--required", type=int, default=3, help="ticks (0.1 s) in a row")
    parser.add_argument("--cool-down", type=int, default=300, help="ticks before a second alert")
    parser.add_argument("--headless", action="store_true", help="no window")
    parser.add_argument("--max-seconds", type=float, help="stop after this much video time")
    parser.add_argument("--no-mirror", action="store_true", help="do not mirror the picture")
    parser.add_argument("--save-dir", type=Path,
                        help="where the save keys write; nothing is stored without it (D-26)")
    return parser


def limit_seconds(
    frames: Iterable[tuple[float, NDArray[np.uint8]]], seconds: float | None
) -> Iterator[tuple[float, NDArray[np.uint8]]]:
    """The frames until `seconds` after the first one (all of them when None)."""
    start: float | None = None
    for time_s, rgb in frames:
        if start is None:
            start = time_s
        if seconds is not None and time_s - start > seconds:
            return
        yield time_s, rgb


def main(argv: Sequence[str] | None = None, detector: Detector | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    run_dir = args.models_dir / args.run
    scorer = Scorer.load(
        args.config or run_dir / "run.toml",
        args.weights or run_dir / "best.pt",
        args.omnifall_dir or args.models_dir / "omnifall",
    )
    pipeline = LivePipeline(scorer, AlertRule(args.threshold, args.required, args.cool_down))

    from ally_fall.live.sources import file_frames, webcam_frames

    frames: Iterable[tuple[float, NDArray[np.uint8]]]
    if args.source.isdigit():
        frames = webcam_frames(int(args.source))
    else:
        frames = file_frames(Path(args.source))
    frames = limit_seconds(frames, args.max_seconds)

    view: Viewer | None = None
    if not args.headless:  # pragma: no cover - needs a screen
        from ally_fall.live.view import CvView

        view = CvView(scorer.run_id, mirror=not args.no_mirror)

    if detector is None:  # pragma: no cover - needs MediaPipe and the model file
        from ally_fall.live.pose import PoseDetector

        pose_model = args.pose_model or args.models_dir / "pose_landmarker_lite.task"
        with PoseDetector(pose_model) as pose:
            summary = run(frames, pose, pipeline, view, args.save_dir)
    else:
        summary = run(frames, detector, pipeline, view, args.save_dir)
    if not args.headless:  # pragma: no cover
        import cv2

        cv2.destroyAllWindows()
    logger.info(
        "done: %d frames, %d ticks, %d scored, %d alerts, %.1f s of video",
        summary.frames, summary.ticks, summary.scored, summary.alerts, summary.seconds_of_video,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
