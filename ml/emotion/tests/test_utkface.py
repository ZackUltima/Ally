import io
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from ally_emotion.paths import FacesRefused
from ally_emotion.utkface import build_utkface, decade, main, parse_name, stratified_split


def test_age_is_parsed() -> None:
    assert parse_name("61_1_0_20170109150557335.jpg.chip.jpg") == 61
    assert parse_name("UTKFace/9_0_4_20170103213057382.jpg.chip.jpg") == 9
    assert parse_name("100_1_2_20170112213615815.jpg") == 100


def test_a_bad_name_is_none() -> None:  # Review Focus 5: never age 0 by default
    for name in ("61_1_20170109.jpg", "x.jpg", "_1_0_20170109150557335.jpg", "61_3_0_2017.jpg"):
        assert parse_name(name) is None, name


def test_decades() -> None:
    assert [decade(a) for a in (0, 9, 10, 59, 60, 79, 80, 116)] == [0, 0, 10, 50, 60, 70, 80, 80]


def test_split_is_stratified_and_repeatable() -> None:
    ages = [5] * 10 + [25] * 50 + [65] * 20 + [85] * 7
    train, test = stratified_split(ages, test_share=0.2, seed=0)
    per_decade = Counter(decade(ages[i]) for i in test)
    assert per_decade == {0: 2, 20: 10, 60: 4, 80: 1}  # 7 x 0.2 = 1.4 -> 1
    again = stratified_split(ages, test_share=0.2, seed=0)
    assert train.tolist() == again[0].tolist() and test.tolist() == again[1].tolist()
    assert stratified_split(ages, seed=1)[1].tolist() != test.tolist()


def test_no_image_is_in_both_parts() -> None:
    ages = list(range(0, 100)) * 3
    train, test = stratified_split(ages)
    assert not set(train.tolist()) & set(test.tolist())
    assert sorted([*train.tolist(), *test.tolist()]) == list(range(len(ages)))


def _archive(path: Path, names: list[str]) -> Path:
    image = pytest.importorskip("PIL.Image")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        for i, name in enumerate(names):
            buf = io.BytesIO()
            image.new("RGB", (200, 200), (i * 20 % 256, 100, 50)).save(buf, format="JPEG")
            info = tarfile.TarInfo(f"UTKFace/{name}")
            info.size = buf.tell()
            buf.seek(0)
            tar.addfile(info, buf)
    return path


def test_60_plus_means_age_60_or_more(tmp_path: Path) -> None:
    ages = (59, 60, 61, 30, 75)
    names = [f"{age}_0_0_2017010915055733{i}.jpg.chip.jpg" for i, age in enumerate(ages)]
    names.append("bad_name.jpg")
    archive = _archive(tmp_path / "raw" / "UTKFace.tar.gz", names)
    counts = build_utkface(archive, tmp_path / "built", tmp_path / "utkface.md", test_share=0.0)
    train = np.load(tmp_path / "built" / "utkface_train.npz")
    assert train["images"].shape == (5, 48, 48) and train["images"].dtype == np.uint8
    assert dict(zip(train["ages"].tolist(), train["is_60plus"].tolist(), strict=True)) == {
        59: False, 60: True, 61: True, 30: False, 75: True
    }
    assert counts["skipped: name does not parse"] == 1 and counts["train"] == 5
    report = (tmp_path / "utkface.md").read_text(encoding="utf-8")
    assert "60+" in report and "name does not parse" in report


def test_a_faces_archive_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLY_FACES_ROOT", str(tmp_path / "faces"))
    with pytest.raises(FacesRefused):
        build_utkface(tmp_path / "faces" / "x.tar.gz", tmp_path / "built", tmp_path / "u.md")


def test_the_command_line_writes_both_parts(tmp_path: Path) -> None:
    names = [f"{20 + i}_1_1_2017010915055733{i}.jpg.chip.jpg" for i in range(10)]
    archive = _archive(tmp_path / "dl" / "UTKFace.tar.gz", names)
    report = tmp_path / "report.md"
    args = ["--data-root", str(tmp_path), "--archive", str(archive), "--report", str(report)]
    assert main(args) == 0
    built = tmp_path / "built" / "emotion"
    sizes = [len(np.load(built / f"utkface_{part}.npz")["ages"]) for part in ("train", "test")]
    assert sizes == [8, 2]
