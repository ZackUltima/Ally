"""Posture as a second reading of the fall model's 16 class probabilities (P1, D-72).

The fall model is not changed: its 16 OmniFall probabilities are added up in five posture
groups instead of only fall plus fallen. Groups are looked up by label name, never by index.
Kneeling, squatting, crawl and jump have no training windows (F3b), so they fall into "other".
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

POSTURES: tuple[str, ...] = ("upright", "sitting", "lying", "changing", "other")
_POSTURE_OF: dict[str, str] = {
    "standing": "upright",
    "walk": "upright",
    "sitting": "sitting",
    "lying": "lying",
    "lie_down": "lying",
    "fallen": "lying",
    "fall": "lying",
    "sit_down": "changing",
    "stand_up": "changing",
    "other": "other",
    "kneel_down": "other",
    "kneeling": "other",
    "squat_down": "other",
    "squatting": "other",
    "crawl": "other",
    "jump": "other",
}


def posture_map(names: tuple[str, ...]) -> NDArray[np.int64]:
    """For each OmniFall label (in the given order), the index of its posture in POSTURES.

    Raises ValueError for a name that is not one of the 16 OmniFall labels, for instance the
    classes of the groups5 label set.
    """
    unknown = [name for name in names if name not in _POSTURE_OF]
    if unknown:
        raise ValueError(
            f"posture needs the 16 OmniFall labels (omnifall16); unknown label names: {unknown}"
        )
    return np.array([POSTURES.index(_POSTURE_OF[name]) for name in names], dtype=np.int64)


def group_probabilities(
    probs: NDArray[np.float32], names: tuple[str, ...]
) -> NDArray[np.float32]:
    """Probabilities (n, 16) added up per posture: (n, 5), rows still sum to the input's."""
    if probs.ndim != 2 or probs.shape[1] != len(names):
        raise ValueError(f"expected probabilities of shape (n, {len(names)}), got {probs.shape}")
    grouping = np.zeros((len(names), len(POSTURES)), dtype=np.float32)
    grouping[np.arange(len(names)), posture_map(names)] = 1.0
    return (probs @ grouping).astype(np.float32)
