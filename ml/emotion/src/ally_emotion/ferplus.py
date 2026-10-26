"""FER+ labels on FER2013 faces (technical plan section E2, T15; D-65 decision 2).

fer2013.csv (emotion, pixels, Usage) and FER+'s fer2013new.csv (Usage, Image name, 8 emotions,
unknown, NF) line up row by row. Splits are FER2013's own: Training -> train, PublicTest -> val,
PrivateTest -> test. A row with no image name or a bad pixel string is skipped and counted, never
shifted onto the next row's label.

Usage: python -m ally_emotion.ferplus --data-root ~/ally-data
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ally_emotion.paths import refuse_faces

logger = logging.getLogger(__name__)

FERPLUS_CLASSES: tuple[str, ...] = (
    "neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"
)
SPLITS: dict[str, str] = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}
SIZE = 48
REPORT = Path(__file__).resolve().parents[2] / "results" / "data" / "ferplus.md"

_VOTE_COLUMNS = len(FERPLUS_CLASSES) + 2  # plus unknown and NF


def majority_label(votes: Sequence[int]) -> int | None:
    """FER+'s majority rule: single votes are outliers; a label wins only with more than half of
    the remaining votes; a win by unknown or not-a-face, or no winner, drops the image."""
    if len(votes) != _VOTE_COLUMNS:
        raise ValueError(f"expected {_VOTE_COLUMNS} vote counts, got {len(votes)}")
    kept = [v if v > 1 else 0 for v in votes]
    total = sum(kept)
    best = max(kept)
    if total == 0 or best * 2 <= total:
        return None
    index = kept.index(best)
    return index if index < len(FERPLUS_CLASSES) else None


def build_ferplus(
    fer2013_csv: Path, ferplus_csv: Path, out_dir: Path, report: Path
) -> dict[str, int]:
    fer_rows = _read(refuse_faces(fer2013_csv))
    plus_rows = _read(refuse_faces(ferplus_csv))
    if len(fer_rows) != len(plus_rows):
        raise ValueError(
            f"{fer2013_csv.name} has {len(fer_rows)} rows "
            f"but {ferplus_csv.name} has {len(plus_rows)}"
        )
    images: dict[str, list[NDArray[np.uint8]]] = {s: [] for s in SPLITS.values()}
    labels: dict[str, list[int]] = {s: [] for s in SPLITS.values()}
    names: dict[str, list[str]] = {s: [] for s in SPLITS.values()}
    counts: dict[str, int] = {}
    per_class = {s: [0] * len(FERPLUS_CLASSES) for s in SPLITS.values()}

    for number, (fer, plus) in enumerate(zip(fer_rows, plus_rows, strict=True), start=2):
        if fer["Usage"] != plus["Usage"]:
            raise ValueError(f"row {number}: Usage {fer['Usage']!r} differs from {plus['Usage']!r}")
        split = SPLITS[fer["Usage"]]
        name = plus["Image name"].strip()
        if not name:
            _count(counts, "skipped: no image name")
            continue
        pixels = _pixels(fer["pixels"])
        if pixels is None:
            _count(counts, "skipped: bad pixels")
            continue
        label = majority_label([int(plus[c]) for c in (*FERPLUS_CLASSES, "unknown", "NF")])
        if label is None:
            _count(counts, f"dropped ({split}): no majority, unknown or not a face")
            continue
        images[split].append(pixels)
        labels[split].append(label)
        names[split].append(name)
        per_class[split][label] += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS.values():
        counts[split] = len(labels[split])
        stack = np.stack(images[split]) if images[split] else np.zeros((0, SIZE, SIZE), np.uint8)
        np.savez_compressed(
            refuse_faces(out_dir / f"ferplus_{split}.npz"),
            images=stack,
            labels=np.array(labels[split], dtype=np.int64),
            names=np.array(names[split], dtype=str),
        )
    _write_report(report, counts, per_class)
    logger.info("FER+ built: %s", counts)
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args(argv)
    root: Path = args.data_root.expanduser()
    build_ferplus(
        root / "raw" / "emotion" / "fer2013" / "fer2013.csv",
        root / "raw" / "emotion" / "ferplus" / "fer2013new.csv",
        root / "built" / "emotion",
        args.report,
    )
    return 0


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _pixels(text: str) -> NDArray[np.uint8] | None:
    parts = text.split()
    if len(parts) != SIZE * SIZE or not all(p.isdigit() for p in parts):
        return None
    values = np.array([int(p) for p in parts], dtype=np.int64)
    if values.max() > 255:
        return None
    return values.astype(np.uint8).reshape(SIZE, SIZE)


def _count(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _write_report(report: Path, counts: dict[str, int], per_class: dict[str, list[int]]) -> None:
    lines = [
        "# FER+ build (EM1a Task 3, D-65)",
        "",
        "Counts only; no image. Splits are FER2013's own (Training, PublicTest, PrivateTest).",
        "",
        "| class | " + " | ".join(per_class) + " |",
        "|---|" + "---|" * len(per_class),
    ]
    for i, name in enumerate(FERPLUS_CLASSES):
        lines.append(f"| {name} | " + " | ".join(str(per_class[s][i]) for s in per_class) + " |")
    lines.append("| **kept** | " + " | ".join(str(counts[s]) for s in per_class) + " |")
    lines += ["", "Left out:", ""]
    other = {k: v for k, v in counts.items() if k not in per_class}
    lines += [f"- {k}: {v}" for k, v in sorted(other.items())] or ["- none"]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())
