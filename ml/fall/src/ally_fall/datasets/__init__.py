"""One small adapter per dataset: which downloaded file is which OmniFall video (D-54)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoItem:
    """A downloaded video file (or frame folder) and its OmniFall identity."""

    source: Path
    video_id: str  # OmniFall `path`
    subject: int
    cam: int
