"""CAUCAFall adapter: downloaded file -> OmniFall identity (real names, Mendeley v5, 2026-10-20)."""

from pathlib import Path

import pytest

from ally_fall.datasets import caucafall


def _touch(root: Path, name: str) -> Path:
    file = root / "CAUCAFall" / name
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(b"")
    return file


def test_items_map_to_omnifall_paths(tmp_path: Path) -> None:
    for name in [
        "Subject.1/Fall backwards/FallBackwardsS1.avi",
        "Subject.2/Fall forward/FallForwardS2.avi",
        "Subject.3/Fall sitting/FallSittingS3.avi",
        "Subject.10/Hop/HopS10.avi",
        "Subject.7/Pick up object/PickupobjectS7.avi",
    ]:
        _touch(tmp_path, name)
    _touch(tmp_path, "Subject.1/Fall backwards/ars100001.png")
    items = caucafall.items(tmp_path)
    assert [(i.video_id, i.subject, i.cam) for i in items] == [
        ("adl/HopS10", 10, 1),
        ("adl/PickupobjectS7", 7, 1),
        ("backwards/FallBackwardsS1", 1, 1),
        ("forward/FallForwardS2", 2, 1),
        ("side/FallSittingS3", 3, 1),
    ]
    assert items[0].source == tmp_path / "CAUCAFall" / "Subject.10" / "Hop" / "HopS10.avi"


def test_unknown_activity_is_an_error(tmp_path: Path) -> None:
    _touch(tmp_path, "Subject.1/Jump/JumpS1.avi")
    with pytest.raises(ValueError, match="JumpS1"):
        caucafall.items(tmp_path)


def test_subject_folder_must_match_file_name(tmp_path: Path) -> None:
    _touch(tmp_path, "Subject.1/Hop/HopS10.avi")
    with pytest.raises(ValueError, match="HopS10"):
        caucafall.items(tmp_path)


def test_dataset_name() -> None:
    assert caucafall.DATASET == "caucafall"
