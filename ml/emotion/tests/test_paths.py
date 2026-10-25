from pathlib import Path

import pytest

from ally_emotion.paths import FacesRefused, data_root, faces_root, refuse_faces


@pytest.fixture
def faces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "faces"
    folder.mkdir()
    (folder / "a.jpg").write_bytes(b"x")
    monkeypatch.setenv("ALLY_FACES_ROOT", str(folder))
    return folder


def test_a_faces_file_is_refused(faces: Path) -> None:
    with pytest.raises(FacesRefused):
        refuse_faces(faces / "a.jpg")


def test_a_link_into_faces_is_refused(faces: Path, tmp_path: Path) -> None:  # Review Focus 2
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "link.jpg").symlink_to(faces / "a.jpg")
    with pytest.raises(FacesRefused):
        refuse_faces(raw / "link.jpg")
    (raw / "dir").symlink_to(faces, target_is_directory=True)
    with pytest.raises(FacesRefused):
        refuse_faces(raw / "dir" / "a.jpg")


def test_dot_dot_into_faces_is_refused(faces: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(FacesRefused):
        refuse_faces(faces / ".." / "faces" / "a.jpg")
    monkeypatch.chdir(faces.parent)
    with pytest.raises(FacesRefused):
        refuse_faces(Path("faces/a.jpg"))


def test_the_faces_folder_itself_is_refused(faces: Path) -> None:
    with pytest.raises(FacesRefused):
        refuse_faces(faces)


def test_other_files_pass(faces: Path, tmp_path: Path) -> None:
    assert refuse_faces(tmp_path / "raw" / "b.jpg") == (tmp_path / "raw" / "b.jpg").resolve()
    other = tmp_path / "faces-other" / "c.jpg"  # a name that only starts with "faces"
    assert refuse_faces(other) == other.resolve()


def test_roots_come_from_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLY_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("ALLY_FACES_ROOT", raising=False)
    assert data_root() == tmp_path
    assert faces_root() == tmp_path / "faces"
    monkeypatch.delenv("ALLY_DATA_ROOT")
    assert data_root() == Path.home() / "ally-data"
