"""OmniFall labels and splits (D-54).

Format checked on 2026-10-19 against the dataset's STRUCTURE.md and LABELS.md
(huggingface.co/datasets/simplexsigil2/omnifall):
- labels/<name>.csv: path, label, start, end, subject, cam, dataset; times in seconds;
  `path` is relative to its dataset and has no file extension.
- labels/label2id.csv: the 16 label names and their ids. Staged datasets use ids 0-9 only.
- splits/cs/<name>/{train,val,test}.csv: one `path` per line.
The labels and splits are CC BY-NC-SA 4.0 (non-commercial, D-19); they are downloaded to
the data root and never stored in the repository.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

from ally_fall.splits import Split
from ally_fall.ticks import Segment

logger = logging.getLogger(__name__)

REPO_ID = "simplexsigil2/omnifall"
# The commit whose split files were checked on 2026-10-20 (D-54; PO: pin it).
REVISION = "83572a37b9e3081df8c06a56874b1d1f2a19386c"
# Overlaps up to one tick are rounding in OmniFall's tables; longer ones are labelling
# conflicts and the video is left out (PO 2026-10-20, option A).
OVERLAP_TOLERANCE_S: float = 0.1
_EPS: float = 1e-9
ALLOW_PATTERNS: list[str] = [
    "labels/*.csv",
    "splits/cs/*/*.csv",
    "LABELS.md",
    "STRUCTURE.md",
    "KNOWN_PITFALLS.md",
    "README.md",
]
COLUMNS: list[str] = ["path", "label", "start", "end", "subject", "cam", "dataset"]
SPLITS: tuple[Split, ...] = ("train", "val", "test")


def download_labels(data_root: Path) -> Path:
    """Download only OmniFall's label, split and documentation files (not its videos)."""
    target = data_root / "omnifall"
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        local_dir=target,
        allow_patterns=ALLOW_PATTERNS,
    )
    return target


def load_table(path: Path) -> pd.DataFrame:
    """Read one label CSV with fixed column types."""
    table = pd.read_csv(
        path,
        dtype={"path": str, "label": "int64", "start": "float64", "end": "float64",
               "subject": "int64", "cam": "int64", "dataset": str},
    )
    missing = set(COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return table[COLUMNS]


def video_identities(path: Path) -> dict[str, tuple[int, int]]:
    """OmniFall `path` -> (subject, cam), for adapters whose file names lack them."""
    table = load_table(path)[["path", "subject", "cam"]].drop_duplicates()
    return {
        str(video): (int(subject), int(cam))
        for video, subject, cam in table.itertuples(index=False)
    }


def label_names(omnifall_dir: Path) -> tuple[str, ...]:
    """The 16 label names in id order, from labels/label2id.csv."""
    table = pd.read_csv(omnifall_dir / "labels" / "label2id.csv").sort_values("id")
    if table["id"].tolist() != list(range(len(table))):
        raise ValueError("label2id.csv ids are not 0..n-1")
    return tuple(table["label"].tolist())


VideoKey = tuple[str, int]  # (path, cam)


def segments_by_video(
    table: pd.DataFrame, dataset: str
) -> tuple[dict[VideoKey, list[Segment]], dict[VideoKey, str]]:
    """Segments per (path, cam) for one dataset, sorted by start, and the left-out videos.

    An overlap up to OVERLAP_TOLERANCE_S is trimmed: the earlier segment ends where the
    later one starts. A longer overlap leaves the video out, with the reason.
    """
    result: dict[VideoKey, list[Segment]] = {}
    conflicts: dict[VideoKey, str] = {}
    rows = table[table["dataset"] == dataset]
    for (path, cam), group in rows.groupby(["path", "cam"], sort=True):
        key = (str(path), int(cam))
        segments = [
            Segment(float(r.start), float(r.end), int(r.label))
            for r in group.sort_values(["start", "end"]).itertuples()
        ]
        largest = max(
            (a.end_s - b.start_s for a, b in zip(segments, segments[1:], strict=False)),
            default=0.0,
        )
        if largest > OVERLAP_TOLERANCE_S + _EPS:
            conflicts[key] = f"overlapping labels ({largest:.2f} s)"
            logger.warning("%s %s cam %d left out: %s", dataset, path, cam, conflicts[key])
            continue
        trimmed = [
            Segment(a.start_s, min(a.end_s, b.start_s), a.label)
            for a, b in zip(segments, segments[1:], strict=False)
        ]
        result[key] = [*trimmed, segments[-1]]
    return result, conflicts


def split_paths(omnifall_dir: Path, split_dir: str) -> dict[Split, set[str]]:
    """Video paths per split from splits/cs/<split_dir>/{train,val,test}.csv."""
    result: dict[Split, set[str]] = {}
    for split in SPLITS:
        file = omnifall_dir / "splits" / "cs" / split_dir / f"{split}.csv"
        result[split] = set(pd.read_csv(file, dtype={"path": str})["path"].dropna())
    return result
