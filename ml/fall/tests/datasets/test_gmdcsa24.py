"""GMDCSA-24 adapter: downloaded file -> OmniFall identity (real names, 2026-10-19)."""

from pathlib import Path

from ally_fall.datasets import gmdcsa24

ROOT = "ekramalam-GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-d3edb5d"


def test_items_map_to_omnifall_paths(tmp_path: Path) -> None:
    for name in ["Subject 1/ADL/01.mp4", "Subject 4/Fall/17.mp4"]:
        file = tmp_path / ROOT / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"")
    (tmp_path / ROOT / "Subject 1" / "ADL.csv").write_text("x")
    items = gmdcsa24.items(tmp_path)
    assert [(i.video_id, i.subject, i.cam) for i in items] == [
        ("Subject_1/ADL/01", 1, 1),
        ("Subject_4/Fall/17", 4, 1),
    ]
    assert items[0].source == tmp_path / ROOT / "Subject 1" / "ADL" / "01.mp4"


def test_dataset_name() -> None:
    assert gmdcsa24.DATASET == "GMDCSA24"
