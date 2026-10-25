import csv
from pathlib import Path

import numpy as np
import pytest

from ally_emotion.ferplus import FERPLUS_CLASSES, build_ferplus, main, majority_label
from ally_emotion.paths import FacesRefused

FER_HEADER = ["emotion", "pixels", "Usage"]
PLUS_HEADER = ["Usage", "Image name", *FERPLUS_CLASSES, "unknown", "NF"]


def _pixels(value: int, n: int = 48 * 48) -> str:
    return " ".join([str(value)] * n)


def _write(path: Path, header: list[str], rows: list[list[str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def _votes(*v: int) -> list[str]:
    return [str(x) for x in v]


def _files(tmp_path: Path, fer: list[list[str]], plus: list[list[str]]) -> tuple[Path, Path]:
    return (
        _write(tmp_path / "raw" / "fer2013.csv", FER_HEADER, fer),
        _write(tmp_path / "raw" / "fer2013new.csv", PLUS_HEADER, plus),
    )


def test_classes_are_ferplus_order() -> None:
    assert FERPLUS_CLASSES == (
        "neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"
    )


def test_no_majority_drops_the_image() -> None:  # 4 of 9 after dropping the 1-vote: not > half
    assert majority_label([4, 0, 0, 1, 3, 2, 0, 0, 0, 0]) is None


def test_single_votes_are_outliers_and_a_majority_wins() -> None:  # 6 of 8: neutral
    assert majority_label([6, 0, 1, 1, 0, 0, 0, 0, 2, 0]) == 0


def test_a_tie_drops_the_image() -> None:  # Review Focus 4
    assert majority_label([5, 5, 0, 0, 0, 0, 0, 0, 0, 0]) is None


def test_unknown_or_not_a_face_drops_the_image() -> None:
    assert majority_label([0, 0, 0, 0, 0, 0, 0, 0, 8, 2]) is None
    assert majority_label([0, 0, 0, 0, 0, 0, 0, 0, 2, 8]) is None


def test_no_votes_left_drops_the_image_and_bad_lengths_are_refused() -> None:
    assert majority_label([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) is None
    with pytest.raises(ValueError):
        majority_label([1, 2, 3])


def test_usage_maps_to_splits(tmp_path: Path) -> None:
    fer, plus = _files(
        tmp_path,
        [
            ["0", _pixels(10), "Training"],
            ["0", _pixels(20), "PublicTest"],
            ["0", _pixels(30), "PrivateTest"],
        ],
        [
            ["Training", "fer0000000.png", *_votes(0, 9, 0, 0, 0, 0, 0, 0, 1, 0)],
            ["PublicTest", "fer0000001.png", *_votes(0, 0, 9, 0, 0, 0, 0, 0, 1, 0)],
            ["PrivateTest", "fer0000002.png", *_votes(0, 0, 0, 0, 0, 0, 0, 10, 0, 0)],
        ],
    )
    counts = build_ferplus(fer, plus, tmp_path / "built", tmp_path / "ferplus.md")
    assert (counts["train"], counts["val"], counts["test"]) == (1, 1, 1)
    test = np.load(tmp_path / "built" / "ferplus_test.npz")
    assert test["images"].shape == (1, 48, 48) and test["images"].dtype == np.uint8
    assert test["labels"].tolist() == [7] and test["names"].tolist() == ["fer0000002.png"]
    assert int(test["images"][0, 0, 0]) == 30


def test_an_empty_name_or_bad_pixels_is_skipped_and_counted(tmp_path: Path) -> None:
    # Review Focus 3
    fer, plus = _files(
        tmp_path,
        [
            ["0", _pixels(1), "Training"],
            ["0", _pixels(2, 100), "Training"],
            ["0", "1 2 x", "Training"],
            ["0", _pixels(256), "Training"],
        ],
        [
            ["Training", "", *_votes(10, 0, 0, 0, 0, 0, 0, 0, 0, 0)],
            ["Training", "fer0000001.png", *_votes(10, 0, 0, 0, 0, 0, 0, 0, 0, 0)],
            ["Training", "fer0000002.png", *_votes(10, 0, 0, 0, 0, 0, 0, 0, 0, 0)],
            ["Training", "fer0000003.png", *_votes(10, 0, 0, 0, 0, 0, 0, 0, 0, 0)],
        ],
    )
    counts = build_ferplus(fer, plus, tmp_path / "built", tmp_path / "ferplus.md")
    assert counts["train"] == 0
    assert counts["skipped: no image name"] == 1 and counts["skipped: bad pixels"] == 3
    report = (tmp_path / "ferplus.md").read_text(encoding="utf-8")
    assert "no image name" in report and "bad pixels" in report


def test_rows_stay_aligned_after_a_skip(tmp_path: Path) -> None:
    fer, plus = _files(
        tmp_path,
        [
            ["0", _pixels(11), "Training"],
            ["0", _pixels(22), "Training"],
            ["0", _pixels(33), "Training"],
        ],
        [
            ["Training", "", *_votes(10, 0, 0, 0, 0, 0, 0, 0, 0, 0)],
            ["Training", "fer0000001.png", *_votes(0, 0, 0, 10, 0, 0, 0, 0, 0, 0)],
            ["Training", "fer0000002.png", *_votes(5, 5, 0, 0, 0, 0, 0, 0, 0, 0)],
        ],
    )
    build_ferplus(fer, plus, tmp_path / "built", tmp_path / "ferplus.md")
    train = np.load(tmp_path / "built" / "ferplus_train.npz")
    assert train["names"].tolist() == ["fer0000001.png"]
    assert train["labels"].tolist() == [3] and int(train["images"][0, 5, 5]) == 22


def test_different_row_counts_stop_the_build(tmp_path: Path) -> None:
    fer, plus = _files(
        tmp_path,
        [["0", _pixels(1), "Training"]],
        [["Training", "a.png", *_votes(10, 0, 0, 0, 0, 0, 0, 0, 0, 0)]] * 2,
    )
    with pytest.raises(ValueError, match="1.*2"):
        build_ferplus(fer, plus, tmp_path / "built", tmp_path / "ferplus.md")


def test_a_usage_mismatch_stops_the_build(tmp_path: Path) -> None:
    fer, plus = _files(
        tmp_path,
        [["0", _pixels(1), "Training"]],
        [["PublicTest", "a.png", *_votes(10, 0, 0, 0, 0, 0, 0, 0, 0, 0)]],
    )
    with pytest.raises(ValueError, match="Usage"):
        build_ferplus(fer, plus, tmp_path / "built", tmp_path / "ferplus.md")


def test_a_faces_path_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLY_FACES_ROOT", str(tmp_path / "raw"))
    fer, plus = _files(tmp_path, [], [])
    with pytest.raises(FacesRefused):
        build_ferplus(fer, plus, tmp_path / "built", tmp_path / "ferplus.md")


def test_the_command_line_uses_the_data_root(tmp_path: Path) -> None:
    fer_csv = tmp_path / "raw" / "emotion" / "fer2013" / "fer2013.csv"
    _write(fer_csv, FER_HEADER, [["0", _pixels(5), "Training"]])
    _write(
        tmp_path / "raw" / "emotion" / "ferplus" / "fer2013new.csv",
        PLUS_HEADER,
        [["Training", "a.png", *_votes(0, 10, 0, 0, 0, 0, 0, 0, 0, 0)]],
    )
    report = tmp_path / "report.md"
    assert main(["--data-root", str(tmp_path), "--report", str(report)]) == 0
    assert (tmp_path / "built" / "emotion" / "ferplus_train.npz").is_file() and report.is_file()
