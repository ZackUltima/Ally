"""prepare_urfall.py / extract_keypoints.py helpers that need neither the network nor a pose model."""

from __future__ import annotations

import csv
import importlib.util
import zipfile
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prep = _load("prepare_urfall")
extract = _load("extract_keypoints")


# ---- prepare_urfall ---------------------------------------------------------------------------------


def test_sequence_selection():
    assert len(prep.sequences("falls")) == 30
    assert len(prep.sequences("adls")) == 40
    assert len(prep.sequences("all")) == 70
    assert prep.sequences("falls")[0] == ("fall-01", "fall")
    assert prep.sequences("adls")[-1] == ("adl-40", "adl")


def test_url_allow_list():
    assert (
        prep._checked_url("fall-01-cam0-rgb.zip")
        == "https://fenix.ur.edu.pl/mkepski/ds/data/fall-01-cam0-rgb.zip"
    )
    with pytest.raises(ValueError, match="refusing"):
        prep._checked_url("https://example.com/evil.zip")
    with pytest.raises(ValueError, match="refusing"):
        prep._checked_url("http://fenix.ur.edu.pl/mkepski/ds/data/x.zip")


def test_write_labels_merges_first_three_columns(tmp_path: Path):
    ann = tmp_path / "annotations"
    ann.mkdir()
    (ann / "urfall-cam0-falls.csv").write_text(
        "fall-01,1,-1,3.1,2.9\nfall-01,2,0,3.3,2.9\nfall-02,1,1,3.1,2.9\n\n", encoding="utf-8"
    )
    (ann / "urfall-cam0-adls.csv").write_text("adl-01,6,-1,6.3\nadl-02,1,-1,6.3\n", encoding="utf-8")
    out = tmp_path / "labels.csv"
    n = prep.write_labels(ann, out, present={"fall-01", "adl-01"})
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert n == 3 and len(rows) == 3
    assert rows[0] == {"sequence": "fall-01", "frame": "1", "label": "-1", "kind": "fall"}
    assert rows[1]["label"] == "0"
    assert rows[2] == {"sequence": "adl-01", "frame": "6", "label": "-1", "kind": "adl"}


def test_extract_frames_flattens_pngs_and_ignores_other_members(tmp_path: Path):
    z = tmp_path / "fall-01-cam0-rgb.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("fall-01-cam0-rgb/fall-01-cam0-rgb-001.png", b"png1")
        zf.writestr("fall-01-cam0-rgb/fall-01-cam0-rgb-002.png", b"png2")
        zf.writestr("fall-01-cam0-rgb/readme.txt", b"ignored")
    out = tmp_path / "rgb" / "fall-01"
    assert prep.extract_frames(z, out) == 2
    assert sorted(p.name for p in out.iterdir()) == ["fall-01-cam0-rgb-001.png", "fall-01-cam0-rgb-002.png"]
    assert (out / "fall-01-cam0-rgb-002.png").read_bytes() == b"png2"


def test_extract_frames_rejects_path_traversal(tmp_path: Path):
    z = tmp_path / "bad.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../../escape.png", b"x")
    out = tmp_path / "rgb" / "bad"
    # basename extraction neutralises the traversal: the file lands inside out_dir
    assert prep.extract_frames(z, out) == 1
    assert (out / "escape.png").exists()
    assert not (tmp_path / "escape.png").exists()


def test_subjects_template_is_never_overwritten(tmp_path: Path):
    path = tmp_path / "subjects.csv"
    assert prep.write_subjects_template(path, ["fall-01", "adl-01"]) is True
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows == [{"sequence": "fall-01", "subject": ""}, {"sequence": "adl-01", "subject": ""}]
    assert prep.subjects_missing(path) is True
    path.write_text("sequence,subject\nfall-01,s1\nadl-01,s2\n", encoding="utf-8")
    assert prep.write_subjects_template(path, ["fall-01"]) is False
    assert prep.subjects_missing(path) is False


def test_download_skips_existing_file(tmp_path: Path):
    dst = tmp_path / "x.csv"
    dst.write_text("cached", encoding="utf-8")
    assert prep.download("urfall-cam0-falls.csv", dst) == dst  # no network call when the file exists
    assert dst.read_text(encoding="utf-8") == "cached"


# ---- extract_keypoints -----------------------------------------------------------------------------


def test_frame_id_parsing():
    assert extract.frame_id_of(Path("fall-01-cam0-rgb-007.png"), 99) == 7
    assert extract.frame_id_of(Path("clip.png"), 4) == 5


def test_iter_sources_finds_videos_and_image_sequences(tmp_path: Path):
    (tmp_path / "clip.mp4").write_bytes(b"")
    seq = tmp_path / "rgb" / "fall-01"
    seq.mkdir(parents=True)
    (seq / "fall-01-cam0-rgb-001.png").write_bytes(b"")
    (tmp_path / "rgb" / "empty").mkdir()
    found = extract.iter_sources(tmp_path)
    assert found == [tmp_path / "clip.mp4", seq]
    assert extract.iter_sources(tmp_path / "clip.mp4") == [tmp_path / "clip.mp4"]
    assert extract.iter_sources(seq) == [seq]


def test_npz_layout_contract_matches_fall_features_input():
    """The extractor's arrays are what fall_features.frame_features consumes."""
    from ally.perception.fall_features import N_KPTS_COCO, frame_features

    assert extract.N_KPTS == N_KPTS_COCO
    kps = np.zeros((3, extract.N_KPTS, 3), np.float32)
    out = frame_features(kps, np.arange(3) / extract.URFALL_FPS, (480, 640), person_present=np.zeros(3, bool))
    assert out.shape == (3, 6) and np.isnan(out).all()
