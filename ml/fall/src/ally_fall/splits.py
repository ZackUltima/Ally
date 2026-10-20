"""OmniFall's cross-subject split for the four step-1 datasets (D-54).

Keys are OmniFall's own `dataset` values. The subject numbers are copied from the
OmniFall paper (D-54) and checked against OmniFall's split files by a local test.
"""

from __future__ import annotations

from typing import Literal

Split = Literal["train", "val", "test"]

CROSS_SUBJECT: dict[str, dict[Split, frozenset[int]]] = {
    "up_fall": {
        "train": frozenset({1, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 17}),
        "val": frozenset({11}),
        "test": frozenset({2, 4, 15, 16}),
    },
    "caucafall": {
        "train": frozenset({1, 2, 3, 4, 5, 6, 7}),
        "val": frozenset({10}),
        "test": frozenset({8, 9}),
    },
    "le2i": {
        "train": frozenset({0, 1, 3, 4, 6, 8}),
        "val": frozenset({5}),
        "test": frozenset({2, 7}),
    },
    "GMDCSA24": {
        "train": frozenset({1, 3}),
        "val": frozenset({2}),
        "test": frozenset({4}),
    },
}


def split_of(dataset: str, subject: int) -> Split:
    """The split a dataset's subject belongs to; KeyError if either is unknown."""
    for split, subjects in CROSS_SUBJECT[dataset].items():
        if subject in subjects:
            return split
    raise KeyError(f"subject {subject} is not in the {dataset} split")
