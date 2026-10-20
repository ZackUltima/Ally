"""Extract keypoints for one downloaded dataset part (D-54, PO decision 3: lite and full).

For each video: run every pose model, save, verify every saved file, and only then delete
the video (when asked). A video whose files are all saved is skipped on a re-run, so an
interrupted run can simply be started again.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ally_fall.build import keypoint_path
from ally_fall.datasets import VideoItem, gmdcsa24
from ally_fall.keypoints import (
    Frame,
    KeypointSeq,
    extract,
    save_seq,
    verify_saved,
    video_frames,
)

logger = logging.getLogger(__name__)


class Extractor(Protocol):
    def __call__(
        self,
        frames: Iterable[Frame],
        model_file: Path,
        *,
        dataset: str,
        video_id: str,
        subject: int,
        cam: int,
        pose_model: str,
    ) -> KeypointSeq: ...


@dataclass
class ExtractReport:
    done: list[str] = field(default_factory=list)
    already_done: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (video_id, reason)


def _video_frames(item: VideoItem) -> Iterable[Frame]:
    return video_frames(item.source)


def run(
    items: Sequence[VideoItem],
    models: Mapping[str, Path],
    data_root: Path,
    dataset: str,
    *,
    delete: bool,
    extractor: Extractor = extract,
    frames_of: Callable[[VideoItem], Iterable[Frame]] = _video_frames,
) -> ExtractReport:
    """Extract, save and verify every item with every model; delete sources only when all pass."""
    report = ExtractReport()
    for item in items:
        outputs = {
            name: keypoint_path(data_root, name, dataset, item.video_id, item.cam)
            for name in models
        }
        if all(path.exists() for path in outputs.values()):
            report.already_done.append(item.video_id)
            continue
        counts: dict[str, int] = {}
        try:
            for name, model_file in models.items():
                try:
                    seq = extractor(
                        frames_of(item), model_file, dataset=dataset, video_id=item.video_id,
                        subject=item.subject, cam=item.cam, pose_model=name,
                    )
                except (OSError, ValueError, RuntimeError) as error:
                    raise ValueError(f"{name}: {error}") from error
                save_seq(seq, outputs[name])
                counts[name] = len(seq.frame_times_s)
            for name, path in outputs.items():
                verify_saved(path, counts[name])
        except ValueError as error:
            logger.error("%s %s failed: %s", dataset, item.video_id, error)
            report.failed.append((item.video_id, str(error)))
            continue
        if delete:
            if item.source.is_dir():
                shutil.rmtree(item.source)
            else:
                item.source.unlink()
            logger.info("deleted %s after verifying %d files", item.source, len(outputs))
        report.done.append(item.video_id)
    return report


ADAPTERS: dict[str, Callable[[Path], list[VideoItem]]] = {gmdcsa24.DATASET: gmdcsa24.items}
MODEL_FILES: dict[str, str] = {
    "lite": "pose_landmarker_lite.task",
    "full": "pose_landmarker_full.task",
}


def main(argv: Sequence[str] | None = None) -> int:
    """python -m ally_fall.extract_dataset --data-root ~/ally-data --dataset GMDCSA24 --raw DIR"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Extract keypoints for one dataset part")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True, choices=sorted(ADAPTERS))
    parser.add_argument("--raw", type=Path, required=True, help="unzipped download")
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--delete", action="store_true", help="delete each video once verified")
    args = parser.parse_args(argv)
    data_root = args.data_root.expanduser()
    models_dir = args.models_dir or data_root / "models"
    models = {name: models_dir / file for name, file in MODEL_FILES.items()}
    items = ADAPTERS[args.dataset](args.raw.expanduser())
    if items and not all(path.exists() for path in models.values()):
        logger.error("pose model files missing in %s", models_dir)
        return 1
    report = run(items, models, data_root, args.dataset, delete=args.delete)
    logger.info(
        "%d done, %d already done, %d failed",
        len(report.done), len(report.already_done), len(report.failed),
    )
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
