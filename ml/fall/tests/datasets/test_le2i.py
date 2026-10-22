"""Le2i adapter: downloaded file -> OmniFall identity (real names, dataUBFC zip, 2026-10-20).

Le2i's file names carry no subject, so it comes from OmniFall's label table.
"""

from pathlib import Path

import pytest

from ally_fall.datasets import le2i

TABLE = """path,label,start,end,subject,cam,dataset
Coffee_room_01/video_14,0,0.0,1.77923,0,1,le2i
Coffee_room_02/video_49,0,0.0,2.0,7,2,le2i
Lecture_room/video_1,0,0.0,3.0,5,5,le2i
Lecture_room/video_1,1,3.0,4.0,5,5,le2i
"""


def _omnifall(tmp_path: Path) -> Path:
    labels = tmp_path / "omnifall" / "labels"
    labels.mkdir(parents=True)
    (labels / "le2i.csv").write_text(TABLE)
    return tmp_path / "omnifall"


def _touch(root: Path, name: str) -> None:
    file = root / name
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(b"")


def test_items_map_to_omnifall_paths(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    for name in [
        "Coffee_room_01/Videos/video (14).avi",
        "Coffee_room_02/Videos/video (49).avi",
        "Lecture room/video (1).avi",
    ]:
        _touch(raw, name)
    _touch(raw, "Coffee_room_01/Annotation_files/video (14).txt")
    items = le2i.items(raw, _omnifall(tmp_path))
    assert [(i.video_id, i.subject, i.cam) for i in items] == [
        ("Coffee_room_01/video_14", 0, 1),
        ("Coffee_room_02/video_49", 7, 2),
        ("Lecture_room/video_1", 5, 5),
    ]
    assert items[2].source == raw / "Lecture room" / "video (1).avi"


def test_video_without_labels_is_an_error(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _touch(raw, "Office/video (99).avi")
    with pytest.raises(ValueError, match="Office/video_99"):
        le2i.items(raw, _omnifall(tmp_path))


def test_unexpected_name_is_an_error(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _touch(raw, "Office/clip1.avi")
    with pytest.raises(ValueError, match="clip1"):
        le2i.items(raw, _omnifall(tmp_path))


def test_dataset_name() -> None:
    assert le2i.DATASET == "le2i"
