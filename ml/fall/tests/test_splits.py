"""Tests for OmniFall's cross-subject split (D-54)."""

import os
from pathlib import Path

import pytest

from ally_fall import omnifall
from ally_fall.splits import CROSS_SUBJECT, split_of

OMNIFALL_DIR = Path(os.environ.get("ALLY_DATA", Path.home() / "ally-data")) / "omnifall"
SPLIT_DIRS = {
    "GMDCSA24": "gmdcsa24",
    "caucafall": "caucafall",
    "le2i": "le2i",
    "up_fall": "up_fall",
}


def test_split_counts() -> None:
    totals = {
        split: sum(len(CROSS_SUBJECT[d][split]) for d in CROSS_SUBJECT)
        for split in ("train", "val", "test")
    }
    assert totals == {"train": 27, "val": 4, "test": 9}


def test_no_subject_in_two_splits() -> None:
    for dataset, splits in CROSS_SUBJECT.items():
        assert not splits["train"] & splits["val"], dataset
        assert not splits["train"] & splits["test"], dataset
        assert not splits["val"] & splits["test"], dataset


def test_split_of() -> None:
    assert split_of("GMDCSA24", 4) == "test"
    assert split_of("up_fall", 11) == "val"
    assert split_of("caucafall", 7) == "train"


def test_unknown_subject() -> None:
    with pytest.raises(KeyError):
        split_of("GMDCSA24", 9)
    with pytest.raises(KeyError):
        split_of("mcfd", 1)


# OmniFall's splits/cs/up_fall/train.csv also lists every video of its val subject (11) and
# of two test subjects (15, 16), so those videos sit in train and in val/test (found
# 2026-10-20, D-54). We split by subject as in the OmniFall paper, so they stay out of
# train; the test checks that the overlap is exactly these subjects and nothing else.
OMNIFALL_TRAIN_OVERLAP: dict[str, frozenset[int]] = {"up_fall": frozenset({11, 15, 16})}


@pytest.mark.skipif(not OMNIFALL_DIR.exists(), reason="OmniFall labels not downloaded")
def test_split_matches_omnifall() -> None:
    for dataset, split_dir in SPLIT_DIRS.items():
        file = "GMDCSA24.csv" if dataset == "GMDCSA24" else f"{dataset}.csv"
        table = omnifall.load_table(OMNIFALL_DIR / "labels" / file)
        subject_of = dict(zip(table["path"], table["subject"], strict=False))
        paths = omnifall.split_paths(OMNIFALL_DIR, split_dir)
        for split, split_paths in paths.items():
            subjects = {int(subject_of[p]) for p in split_paths}
            expected = set(CROSS_SUBJECT[dataset][split])
            if split == "train":
                expected |= OMNIFALL_TRAIN_OVERLAP.get(dataset, frozenset())
            assert subjects == expected, (dataset, split)
        overlap = paths["train"] & (paths["val"] | paths["test"])
        overlap_subjects = {int(subject_of[p]) for p in overlap}
        assert overlap_subjects == set(OMNIFALL_TRAIN_OVERLAP.get(dataset, frozenset())), dataset
