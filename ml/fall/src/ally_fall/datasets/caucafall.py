"""CAUCAFall (Mendeley Data 7w7fccy7ky v5, CC BY 4.0): `Subject.N/<activity>/<Name>SN.avi`.

OmniFall names these videos `<group>/<Name>SN`, one camera (checked 2026-10-20 on all 100).
"""

from __future__ import annotations

import re
from pathlib import Path

from ally_fall.datasets import VideoItem

DATASET = "caucafall"
_PATTERN = re.compile(r"Subject\.(\d+)/[^/]+/([A-Za-z]+)S(\d+)\.avi$")
_GROUPS: dict[str, str] = {
    "FallBackwards": "backwards",
    "FallForward": "forward",
    "FallLeft": "side",
    "FallRight": "side",
    "FallSitting": "side",
    "Hop": "adl",
    "Kneel": "adl",
    "Pickupobject": "adl",
    "SitDown": "adl",
    "Walk": "adl",
}


def items(raw_dir: Path) -> list[VideoItem]:
    """Every video under the download, sorted by OmniFall path; unknown names are an error."""
    found: list[VideoItem] = []
    for file in raw_dir.rglob("*.avi"):
        match = _PATTERN.search(file.relative_to(raw_dir).as_posix())
        if match is None:
            raise ValueError(f"unexpected CAUCAFall file: {file}")
        folder_subject, name, file_subject = match.groups()
        if name not in _GROUPS:
            raise ValueError(f"unknown CAUCAFall activity: {file.name}")
        if folder_subject != file_subject:
            raise ValueError(f"subject folder does not match file name: {file}")
        found.append(
            VideoItem(source=file, video_id=f"{_GROUPS[name]}/{name}S{file_subject}",
                      subject=int(file_subject), cam=1)
        )
    return sorted(found, key=lambda item: item.video_id)
