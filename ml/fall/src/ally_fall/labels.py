"""The two label sets the fall models are trained on (D-54, D-55, D-62).

omnifall16 keeps OmniFall's 16 labels in its own order; groups5 is D-54's four groups
with standing and walk split out as upright (technical plan T10), so "off the ground"
still works. Groups are looked up by label name, never by a retyped index.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

LabelSet = Literal["omnifall16", "groups5"]

GROUPS5: tuple[str, ...] = ("fall", "fallen", "lying", "upright", "rest")
_GROUP_OF: dict[str, str] = {
    "fall": "fall",
    "fallen": "fallen",
    "lie_down": "lying",
    "lying": "lying",
    "standing": "upright",
    "walk": "upright",
}


def class_names(label_set: LabelSet, names: tuple[str, ...]) -> tuple[str, ...]:
    """The class names of a label set, in class-index order."""
    if label_set == "omnifall16":
        return names
    if label_set == "groups5":
        return GROUPS5
    raise ValueError(f"unknown label set {label_set!r}")


def label_map(names: tuple[str, ...], label_set: LabelSet) -> NDArray[np.int64]:
    """For each OmniFall label index, its class index in the label set."""
    classes = class_names(label_set, names)
    if label_set == "omnifall16":
        return np.arange(len(names), dtype=np.int64)
    return np.array([classes.index(_GROUP_OF.get(name, "rest")) for name in names], dtype=np.int64)


def fall_classes(label_set: LabelSet, names: tuple[str, ...]) -> tuple[int, ...]:
    """The class indices of the fall class, fall and fallen (D-55)."""
    classes = class_names(label_set, names)
    return (classes.index("fall"), classes.index("fallen"))
