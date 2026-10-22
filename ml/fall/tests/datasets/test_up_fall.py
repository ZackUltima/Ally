"""UP-Fall adapter: downloaded zip -> OmniFall identity, and the official page -> download links.

Real names from the HAR-UP page and Subject1Activity1Trial1Camera1.zip (2026-10-20).
"""

import html
from pathlib import Path

import pytest

from ally_fall.datasets import up_fall

TABLE = """path,label,start,end,subject,cam,dataset
Subject1/Activity1/Trial1/Subject1Activity1Trial1Camera1,8,0.0,2.89409,1,1,up_fall
Subject1/Activity1/Trial1/Subject1Activity1Trial1Camera1,1,2.894,4.994,1,1,up_fall
Subject16/Activity11/Trial3/Subject16Activity11Trial3Camera2,9,0.0,60.0,16,2,up_fall
Subject2/Activity3/Trial2/Subject2Activity3Trial2Camera1,8,0.0,10.0,5,1,up_fall
"""

PAGE = """<div id="Subject1Activity1Trial1" class="tabcontent2">
<h5>Subject1Activity1Trial1</h5>
<p><a href="https://drive.google.com/a/up.edu.mx/uc?id=1-qsLb3ubU_MZ5-TO3e-AAg89cgBzEPZX&export=download">DataSet</a></p>
<p><a href="https://drive.google.com/a/up.edu.mx/uc?id=1x-gpsGcP1jMAvWAZ9oM1O7MsGW8VaCma&export=download">Camera1</a>
<a href="https://drive.google.com/a/up.edu.mx/uc?id=1Lh9EMMbDu1Z8yooJVXc4FeAHrOFuR2NL&export=download">Camera2</a></p>
<p><a href="https://drive.google.com/a/up.edu.mx/uc?id=16dPrxHv3lXo9nYl-pvdLoOAaF849JWyK&export=download">Camera1_OF</a></p>
</div>
<div id="Subject8Activity11Trial3" class="tabcontent2">
<h5>Subject8Activity11Trial3</h5>
<p><a href="https://drive.google.com/a/up.edu.mx/uc?id=AAA&export=download">DataSet</a></p>
</div>"""


def _omnifall(tmp_path: Path) -> Path:
    labels = tmp_path / "omnifall" / "labels"
    labels.mkdir(parents=True)
    (labels / "up_fall.csv").write_text(TABLE)
    return tmp_path / "omnifall"


def _touch(root: Path, name: str) -> Path:
    file = root / name
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(b"")
    return file


def test_items_map_to_omnifall_paths(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _touch(raw, "Subject16Activity11Trial3Camera2.zip")
    _touch(raw, "Subject1Activity1Trial1Camera1.zip")
    _touch(raw, "Subject1Activity1Trial1Camera2.zip.part")  # a download still in progress
    items = up_fall.items(raw, _omnifall(tmp_path))
    # Sorted by number (subject, activity, trial, camera), not as text.
    assert [(i.video_id, i.subject, i.cam) for i in items] == [
        ("Subject1/Activity1/Trial1/Subject1Activity1Trial1Camera1", 1, 1),
        ("Subject16/Activity11/Trial3/Subject16Activity11Trial3Camera2", 16, 2),
    ]
    assert items[0].source == raw / "Subject1Activity1Trial1Camera1.zip"


def test_zip_without_labels_is_an_error(tmp_path: Path) -> None:
    _touch(tmp_path / "raw", "Subject3Activity1Trial1Camera1.zip")
    with pytest.raises(ValueError, match="Subject3Activity1Trial1Camera1"):
        up_fall.items(tmp_path / "raw", _omnifall(tmp_path))


def test_subject_that_differs_from_omnifall_is_an_error(tmp_path: Path) -> None:
    _touch(tmp_path / "raw", "Subject2Activity3Trial2Camera1.zip")
    with pytest.raises(ValueError, match="subject"):
        up_fall.items(tmp_path / "raw", _omnifall(tmp_path))


def test_unexpected_zip_name_is_an_error(tmp_path: Path) -> None:
    _touch(tmp_path / "raw", "Subject1Activity1Trial1Camera1_OF.zip")
    with pytest.raises(ValueError, match="_OF"):
        up_fall.items(tmp_path / "raw", _omnifall(tmp_path))


def test_camera_links_from_the_official_page() -> None:
    expected = {
        "Subject1/Activity1/Trial1/Subject1Activity1Trial1Camera1":
            "1x-gpsGcP1jMAvWAZ9oM1O7MsGW8VaCma",
        "Subject1/Activity1/Trial1/Subject1Activity1Trial1Camera2":
            "1Lh9EMMbDu1Z8yooJVXc4FeAHrOFuR2NL",
    }
    assert up_fall.camera_links(PAGE) == expected
    # The live page carries this markup escaped twice inside an embed attribute.
    assert up_fall.camera_links(html.escape(html.escape(PAGE))) == expected


def test_camera_link_listed_twice_with_different_files_is_an_error() -> None:
    twice = PAGE + PAGE.replace("1x-gpsGcP1jMAvWAZ9oM1O7MsGW8VaCma", "other")
    with pytest.raises(ValueError, match="Subject1Activity1Trial1Camera1"):
        up_fall.camera_links(twice)


def test_dataset_name() -> None:
    assert up_fall.DATASET == "up_fall"
