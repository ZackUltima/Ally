"""Data locations and the FACES guard (technical plan section E2, D-41).

FACES is a test set only: no training, validation, model-choice or build code may read a path
under the FACES root. Links and relative paths are resolved before the check, so a link inside
~/ally-data/raw that points into FACES is refused like a direct path.
"""

from __future__ import annotations

import os
from pathlib import Path


class FacesRefused(Exception):
    """A path lies under the FACES root, which only the evaluation code may read (D-41)."""


def data_root() -> Path:
    return Path(os.environ.get("ALLY_DATA_ROOT", str(Path.home() / "ally-data")))


def faces_root() -> Path:
    return Path(os.environ.get("ALLY_FACES_ROOT", str(data_root() / "faces")))


def refuse_faces(path: Path) -> Path:
    """The resolved path; raises FacesRefused when it lies under the FACES root."""
    resolved = path.resolve()
    if resolved.is_relative_to(faces_root().resolve()):
        raise FacesRefused(f"{path} lies under the FACES root; FACES is a test set only (D-41)")
    return resolved
