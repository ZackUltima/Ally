"""Build the training, validation and test data from saved keypoints (technical plan §2).

Each split is saved once as ticks (not as repeated windows): features [sum T, 33, 3]
(x, y, visibility), labels [sum T], per-video offsets, and the index of every usable
window's last tick (window_end) with its label (y). A window never crosses two videos.

Usage: python -m ally_fall.build --data-root ~/ally-data --datasets GMDCSA24 [--pose-model lite]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ally_fall.keypoints import POINTS, KeypointSeq, load_seq
from ally_fall.omnifall import label_names, load_table, segments_by_video
from ally_fall.splits import Split, split_of
from ally_fall.ticks import (
    Segment,
    frame_labels,
    latest_frame_index,
    mark_missing,
    nearest_frame_index,
    tick_times,
)
from ally_fall.windows import WINDOW_TICKS

logger = logging.getLogger(__name__)

LABEL_FILES: dict[str, str] = {
    "GMDCSA24": "GMDCSA24.csv",
    "caucafall": "caucafall.csv",
    "le2i": "le2i.csv",
    "up_fall": "up_fall.csv",
}
SPLITS: tuple[Split, ...] = ("train", "val", "test")

# Videos left out by name, with the reason reported in BuildReport.skipped. UP-Fall camera 1 of
# Activity 11 (lying) for subjects 5, 11, 12 and 15: the person lies at the bottom edge of the
# picture and a pose is found in only 0-5% of frames (D-54, PO 2026-10-22).
_UP_FALL_OUT_OF_PICTURE = "person out of camera 1's picture, no pose (D-54)"
EXCLUDED: dict[tuple[str, str, int], str] = {
    ("up_fall", f"Subject{s}/Activity11/Trial{t}/Subject{s}Activity11Trial{t}Camera1", 1): (
        _UP_FALL_OUT_OF_PICTURE
    )
    for s in (5, 11, 12, 15)
    for t in (1, 2, 3)
}
# Whole cameras left out. UP-Fall camera 2: a seated bystander is in view in most sessions, and
# once the person is on the floor the pose is often on the bystander until the clip ends (drawn
# for 6 subjects; 17% of its clips jump; 15.8% of training fall windows). Camera 1 films every
# trial and is clean (D-54, PO 2026-10-22, option A).
EXCLUDED_CAMERAS: dict[tuple[str, int], str] = {
    ("up_fall", 2): "camera 2, pose often on a seated bystander (D-54)",
}


@dataclass
class SplitStats:
    videos: int = 0
    ticks: int = 0
    windows: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class BuildReport:
    splits: dict[Split, SplitStats]
    skipped: list[tuple[str, str, str]]  # (dataset, video, reason)


@dataclass
class _SplitData:
    features: list[NDArray[np.float32]] = field(default_factory=list)
    labels: list[NDArray[np.int64]] = field(default_factory=list)
    subject: list[int] = field(default_factory=list)
    dataset: list[str] = field(default_factory=list)
    video_id: list[str] = field(default_factory=list)


def keypoint_path(data_root: Path, pose_model: str, dataset: str, video_id: str, cam: int) -> Path:
    """Where a video's keypoints are saved: keypoints/<pose_model>/<dataset>/<id>__cam<cam>.npz."""
    name = video_id.replace("/", "__") + f"__cam{cam}.npz"
    return data_root / "keypoints" / pose_model / dataset / name


def video_ticks(
    seq: KeypointSeq, segments: list[Segment]
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """One video on the 10 fps grid: features [T, 33, 3] and labels [T] (-1 in label gaps)."""
    ticks = tick_times(seq.frame_times_s)
    latest = latest_frame_index(seq.frame_times_s, ticks)
    xy, vis = mark_missing(seq.xy[latest], seq.vis[latest])
    features = np.concatenate([xy, vis[..., None]], axis=-1).astype(np.float32)
    per_frame = frame_labels(seq.frame_times_s, segments)
    labels = per_frame[nearest_frame_index(seq.frame_times_s, ticks)]
    return features, labels


def window_features(
    features: NDArray[np.float32], window_end: NDArray[np.int64], length: int = WINDOW_TICKS
) -> NDArray[np.float32]:
    """Gather [N, length, 33, 3] windows from tick features by each window's last tick."""
    index = window_end[:, None] - (length - 1) + np.arange(length)[None, :]
    return features[index]


def _save_split(path: Path, data: _SplitData) -> SplitStats:
    lengths = np.array([len(f) for f in data.features], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    starts = starts[: len(lengths)]
    features = (
        np.concatenate(data.features)
        if data.features
        else np.zeros((0, POINTS, 3), dtype=np.float32)
    )
    labels = np.concatenate(data.labels) if data.labels else np.zeros((0,), dtype=np.int64)
    ends: list[NDArray[np.int64]] = []
    for start, length in zip(starts, lengths, strict=True):
        local = np.arange(WINDOW_TICKS - 1, length, dtype=np.int64)
        local = local[labels[start + local] != -1]
        ends.append(start + local)
    window_end = np.concatenate(ends) if ends else np.zeros((0,), dtype=np.int64)
    y = labels[window_end]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        features=features,
        labels=labels,
        video_start=starts,
        video_len=lengths,
        subject=np.array(data.subject, dtype=np.int64),
        dataset=np.array(data.dataset, dtype=str),
        video_id=np.array(data.video_id, dtype=str),
        window_end=window_end,
        y=y,
    )
    return SplitStats(videos=len(lengths), ticks=int(lengths.sum()), windows=len(window_end))


def build(
    data_root: Path,
    datasets: Sequence[str],
    pose_model: str = "lite",
    excluded: Mapping[tuple[str, str, int], str] = EXCLUDED,
    excluded_cameras: Mapping[tuple[str, int], str] = EXCLUDED_CAMERAS,
) -> BuildReport:
    """Build built/<pose_model>/{train,val,test}.npz and the training label counts."""
    omnifall_dir = data_root / "omnifall"
    names = label_names(omnifall_dir)
    split_data: dict[Split, _SplitData] = {split: _SplitData() for split in SPLITS}
    skipped: list[tuple[str, str, str]] = []

    for dataset in datasets:
        table = load_table(omnifall_dir / "labels" / LABEL_FILES[dataset])
        segments, conflicts = segments_by_video(table, dataset)
        found: set[tuple[str, int]] = set()
        folder = data_root / "keypoints" / pose_model / dataset
        for file in sorted(folder.glob("*.npz")):
            seq = load_seq(file)
            key = (seq.video_id, seq.cam)
            if key in conflicts:
                skipped.append((dataset, seq.video_id, conflicts[key]))
                continue
            if key not in segments:
                skipped.append((dataset, seq.video_id, "no labels"))
                continue
            found.add(key)
            if (dataset, *key) in excluded:
                skipped.append((dataset, seq.video_id, excluded[(dataset, *key)]))
                continue
            if (dataset, seq.cam) in excluded_cameras:
                skipped.append((dataset, seq.video_id, excluded_cameras[(dataset, seq.cam)]))
                continue
            try:
                split = split_of(dataset, seq.subject)
            except KeyError:
                skipped.append((dataset, seq.video_id, f"subject {seq.subject} not in the split"))
                continue
            features, labels = video_ticks(seq, segments[key])
            data = split_data[split]
            data.features.append(features)
            data.labels.append(labels)
            data.subject.append(seq.subject)
            data.dataset.append(dataset)
            data.video_id.append(seq.video_id)
        for path, _cam in sorted(set(segments) - found):
            skipped.append((dataset, path, "no keypoints"))

    out = data_root / "built" / pose_model
    stats: dict[Split, SplitStats] = {}
    for split in SPLITS:
        stats[split] = _save_split(out / f"{split}.npz", split_data[split])
        with np.load(out / f"{split}.npz", allow_pickle=False) as saved:
            y = saved["y"]
        stats[split].label_counts = {name: int((y == i).sum()) for i, name in enumerate(names)}
    (out / "label_counts_train.json").write_text(
        json.dumps(stats["train"].label_counts, indent=2) + "\n", encoding="utf-8"
    )
    for dataset, video, reason in skipped:
        logger.warning("skipped %s %s: %s", dataset, video, reason)
    return BuildReport(splits=stats, skipped=skipped)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True, choices=sorted(LABEL_FILES))
    parser.add_argument("--pose-model", default="lite", choices=["lite", "full"])
    args = parser.parse_args(argv)
    try:
        report = build(args.data_root.expanduser(), args.datasets, args.pose_model)
    except (OSError, ValueError, KeyError) as error:
        logger.error("build failed: %s", error)
        return 1
    for split, s in report.splits.items():
        logger.info("%s: %d videos, %d ticks, %d windows", split, s.videos, s.ticks, s.windows)
    logger.info("%d skipped", len(report.skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
