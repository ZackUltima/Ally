"""UTKFace for the age classifier (technical plan section E2; D-65 decisions 1 and 4).

Ages come from the file names, `[age]_[gender]_[race]_[date].jpg[.chip.jpg]`; a name that does
not parse is skipped and counted, never given age 0. The split is 80% training and 20% held-out
test, stratified by decade of age (80+ as one group), seed 0; "60+" means age 60 or more. Each
image goes through crop_face with the whole image as the box and margin 0, the same path as
every other face. The archive is read in memory; nothing is extracted to disk.

Usage: python -m ally_emotion.utkface --data-root ~/ally-data --archive <UTKFace.tar.gz>
"""

from __future__ import annotations

import argparse
import io
import logging
import math
import re
import sys
import tarfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ally_emotion.crop import Box, crop_face
from ally_emotion.paths import refuse_faces

logger = logging.getLogger(__name__)

AGE_60_PLUS = 60
TOP_DECADE = 80
REPORT = Path(__file__).resolve().parents[2] / "results" / "data" / "utkface.md"

_NAME = re.compile(r"^(?P<age>\d{1,3})_[01]_[0-4]_\d+\.jpg(?:\.chip\.jpg)?$")
_SKIP = "skipped: name does not parse"


def parse_name(name: str) -> int | None:
    m = _NAME.match(Path(name).name)
    return int(m.group("age")) if m else None


def decade(age: int) -> int:
    return min(age // 10 * 10, TOP_DECADE)


def stratified_split(
    ages: Sequence[int], test_share: float = 0.2, seed: int = 0
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Training and test indices; each decade gives round-half-up(n x share) to the test."""
    rng = np.random.default_rng(seed)
    groups: dict[int, list[int]] = {}
    for i, age in enumerate(ages):
        groups.setdefault(decade(age), []).append(i)
    train: list[int] = []
    test: list[int] = []
    for d in sorted(groups):
        members = np.array(groups[d], dtype=np.int64)
        rng.shuffle(members)
        n_test = math.floor(len(members) * test_share + 0.5)
        test.extend(members[:n_test].tolist())
        train.extend(members[n_test:].tolist())
    return np.array(sorted(train), dtype=np.int64), np.array(sorted(test), dtype=np.int64)


def build_utkface(
    archive: Path, out_dir: Path, report: Path, test_share: float = 0.2, seed: int = 0
) -> dict[str, int]:
    from PIL import Image  # image reading only (EM1a tech stack)

    images: list[NDArray[np.uint8]] = []
    ages: list[int] = []
    counts: dict[str, int] = {}
    with tarfile.open(refuse_faces(archive), "r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".jpg"):
                continue
            age = parse_name(member.name)
            if age is None:
                counts[_SKIP] = counts.get(_SKIP, 0) + 1
                continue
            data = tar.extractfile(member)
            if data is None:
                raise ValueError(f"{archive}: cannot read {member.name}")
            rgb = np.asarray(Image.open(io.BytesIO(data.read())).convert("RGB"), dtype=np.uint8)
            images.append(crop_face(rgb, Box(0, 0, rgb.shape[1], rgb.shape[0]), 0.0))
            ages.append(age)

    train, test = stratified_split(ages, test_share, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_images = np.stack(images) if images else np.zeros((0, 48, 48), np.uint8)
    all_ages = np.array(ages, dtype=np.int64)
    parts = {"train": train, "test": test}
    for part, idx in parts.items():
        counts[part] = len(idx)
        np.savez_compressed(
            refuse_faces(out_dir / f"utkface_{part}.npz"),
            images=all_images[idx],
            ages=all_ages[idx],
            is_60plus=all_ages[idx] >= AGE_60_PLUS,
        )
    _write_report(report, all_ages, parts, counts)
    logger.info("UTKFace built: %s", counts)
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args(argv)
    root: Path = args.data_root.expanduser()
    archive: Path = args.archive or root / "raw" / "emotion" / "utkface" / "UTKFace.tar.gz"
    build_utkface(archive.expanduser(), root / "built" / "emotion", args.report)
    return 0


def _write_report(
    report: Path,
    ages: NDArray[np.int64],
    parts: dict[str, NDArray[np.int64]],
    counts: dict[str, int],
) -> None:
    decades = sorted({decade(int(a)) for a in ages})
    lines = [
        "# UTKFace build (EM1a Task 4, D-65)",
        "",
        "Counts only; no image. Split stratified by decade of age, seed 0.",
        "",
        "| decade | " + " | ".join(parts) + " |",
        "|---|" + "---|" * len(parts),
    ]
    for d in decades:
        label = f"{d}+" if d == TOP_DECADE else f"{d}-{d + 9}"
        cells = [str(int(sum(decade(int(a)) == d for a in ages[idx]))) for idx in parts.values()]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    shares = []
    for idx in parts.values():
        share = float(np.mean(ages[idx] >= AGE_60_PLUS)) if len(idx) else 0.0
        shares.append(f"{share:.1%}")
    lines.append("| **60+ share** | " + " | ".join(shares) + " |")
    lines += ["", f"- {_SKIP.removeprefix('skipped: ')}: {counts.get(_SKIP, 0)}"]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())
