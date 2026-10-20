"""GMDCSA-24 (Zenodo 12921216 v2.0, CC BY 4.0): `Subject N/<ADL|Fall>/NN.mp4`, one camera.

OmniFall names these videos `Subject_N/<ADL|Fall>/NN` (checked 2026-10-19).
"""

from __future__ import annotations

import re
from pathlib import Path

from ally_fall.datasets import VideoItem

DATASET = "GMDCSA24"
_PATTERN = re.compile(r"Subject (\d+)/(ADL|Fall)/(\d+)\.mp4$")


def items(raw_dir: Path) -> list[VideoItem]:
    """Every video under the unzipped download, sorted by OmniFall path."""
    found: list[VideoItem] = []
    for file in raw_dir.rglob("*.mp4"):
        match = _PATTERN.search(file.relative_to(raw_dir).as_posix())
        if match is None:
            continue
        subject, kind, number = match.groups()
        found.append(
            VideoItem(source=file, video_id=f"Subject_{subject}/{kind}/{number}",
                      subject=int(subject), cam=1)
        )
    return sorted(found, key=lambda item: item.video_id)
