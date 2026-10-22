"""Le2i (dataUBFC 10.25666/DATAUBFC-2024-04-09, CC BY-NC-SA 3.0): `<scene>/[Videos/]video (N).avi`.

OmniFall names these videos `<scene>/video_N` (spaces in scene names become `_`), one camera
per scene (checked 2026-10-20 on all 190). The file names carry no subject, so the subject
and camera come from OmniFall's label table.
"""

from __future__ import annotations

import re
from pathlib import Path

from ally_fall.datasets import VideoItem
from ally_fall.omnifall import video_identities

DATASET = "le2i"
_PATTERN = re.compile(r"([^/]+)/(?:Videos/)?video \((\d+)\)\.avi$")


def items(raw_dir: Path, omnifall_dir: Path) -> list[VideoItem]:
    """Every video under the unpacked download, sorted by OmniFall path."""
    identity = video_identities(omnifall_dir / "labels" / "le2i.csv")
    found: list[VideoItem] = []
    for file in raw_dir.rglob("*.avi"):
        match = _PATTERN.search(file.relative_to(raw_dir).as_posix())
        if match is None:
            raise ValueError(f"unexpected Le2i file: {file}")
        scene, number = match.groups()
        video_id = f"{scene.replace(' ', '_')}/video_{number}"
        if video_id not in identity:
            raise ValueError(f"no OmniFall labels for {video_id} ({file})")
        subject, cam = identity[video_id]
        found.append(VideoItem(source=file, video_id=video_id, subject=subject, cam=cam))
    return sorted(found, key=lambda item: item.video_id)
