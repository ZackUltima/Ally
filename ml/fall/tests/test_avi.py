"""Tests for the uncompressed (24-bit DIB) AVI reader (Le2i, 2026-10-20)."""

import struct
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from ally_fall.avi import dib_frames, is_uncompressed_avi


def _chunk(fourcc: bytes, data: bytes) -> bytes:
    return fourcc + struct.pack("<I", len(data)) + data + (b"\0" if len(data) % 2 else b"")


def _list(kind: bytes, body: bytes) -> bytes:
    return _chunk(b"LIST", kind + body)


def _dib(rgb: NDArray[np.uint8], bottom_up: bool) -> bytes:
    """RGB image -> BGR rows padded to 4 bytes, bottom row first when bottom_up."""
    height, width, _ = rgb.shape
    row = ((width * 24 + 31) // 32) * 4
    out = np.zeros((height, row), dtype=np.uint8)
    out[:, : width * 3] = rgb[..., ::-1].reshape(height, width * 3)
    return (out[::-1] if bottom_up else out).tobytes()


def _write_avi(
    path: Path,
    images: list[NDArray[np.uint8]],
    *,
    scale: int = 1,
    rate: int = 25,
    bits: int = 24,
    compression: int = 0,
    bottom_up: bool = True,
    grouped: bool = False,
    extra: bytes = b"",
) -> Path:
    height, width, _ = images[0].shape
    strh_vids = b"vids" + b"\0" * 4 + struct.pack("<IHHIIIII", 0, 0, 0, 0, scale, rate, 0,
                                                  len(images)) + b"\0" * 24
    strf_vids = struct.pack("<IiiHHIIiiII", 40, width, height if bottom_up else -height, 1, bits,
                            compression, 0, 0, 0, 0, 0)
    strh_auds = b"auds" + b"\0" * 52
    strf_auds = struct.pack("<HHIIHH", 1, 1, 8000, 8000, 1, 8)
    hdrl = _list(b"hdrl", _chunk(b"avih", b"\0" * 56)
                 + _list(b"strl", _chunk(b"strh", strh_vids) + _chunk(b"strf", strf_vids))
                 + _list(b"strl", _chunk(b"strh", strh_auds) + _chunk(b"strf", strf_auds)))
    movi = b""
    for image in images:
        pair = _chunk(b"00db", _dib(image, bottom_up)) + _chunk(b"01wb", b"\x80" * 7)
        movi += _list(b"rec ", pair) if grouped else pair
    body = b"AVI " + hdrl + _list(b"movi", movi) + _chunk(b"idx1", b"")
    path.write_bytes(_chunk(b"RIFF", body) + extra)
    return path


def _image(height: int = 3, width: int = 5, value: int = 0) -> NDArray[np.uint8]:
    image = np.full((height, width, 3), value, dtype=np.uint8)
    image[0, 0] = (255, 0, 0)  # top-left pixel red
    return image


def test_frames_are_upright_rgb(tmp_path: Path) -> None:
    frames = list(dib_frames(_write_avi(tmp_path / "a.avi", [_image(), _image(value=9)])))
    assert len(frames) == 2
    first = frames[0][1]
    assert first.shape == (3, 5, 3) and first.dtype == np.uint8
    assert tuple(first[0, 0]) == (255, 0, 0)
    assert tuple(first[2, 4]) == (0, 0, 0)
    assert tuple(frames[1][1][1, 1]) == (9, 9, 9)


def test_top_down_frames_are_not_flipped(tmp_path: Path) -> None:
    path = _write_avi(tmp_path / "a.avi", [_image()], bottom_up=False)
    assert tuple(next(dib_frames(path))[1][0, 0]) == (255, 0, 0)


def test_frames_inside_rec_lists(tmp_path: Path) -> None:
    path = _write_avi(tmp_path / "a.avi", [_image(), _image(), _image()], grouped=True)
    assert len(list(dib_frames(path))) == 3


def test_times_come_from_scale_and_rate(tmp_path: Path) -> None:
    path = _write_avi(tmp_path / "a.avi", [_image()] * 4, scale=41666, rate=1000000)
    times = [t for t, _ in dib_frames(path)]
    assert times == pytest.approx([i * 41666 / 1000000 for i in range(4)])


def test_recognises_only_uncompressed_avi(tmp_path: Path) -> None:
    assert is_uncompressed_avi(_write_avi(tmp_path / "a.avi", [_image()]))
    mjpg = tmp_path / "m.avi"
    writer = cv2.VideoWriter(str(mjpg), cv2.VideoWriter_fourcc(*"MJPG"), 25, (16, 16))
    writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    writer.release()
    assert not is_uncompressed_avi(mjpg)
    (tmp_path / "x.mp4").write_bytes(b"\0\0\0\x18ftypmp42")
    assert not is_uncompressed_avi(tmp_path / "x.mp4")


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"bits": 32}, "24-bit"),
        ({"compression": 1}, "uncompressed"),
        ({"extra": b"RIFF\x04\0\0\0AVIX"}, "OpenDML"),
    ],
)
def test_rejects_unsupported(tmp_path: Path, options: dict[str, object], message: str) -> None:
    path = _write_avi(tmp_path / "a.avi", [_image()], **options)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=message):
        list(dib_frames(path))


def test_video_frames_uses_this_reader_for_uncompressed_avi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ally_fall import keypoints

    def no_opencv(*_: object) -> None:
        raise AssertionError("OpenCV must not decode uncompressed AVI")

    monkeypatch.setattr(keypoints.cv2, "VideoCapture", no_opencv)
    path = _write_avi(tmp_path / "a.avi", [_image(), _image()], scale=1, rate=24)
    frames = list(keypoints.video_frames(path))
    assert [t for t, _ in frames] == pytest.approx([0.0, 1 / 24])
    assert tuple(frames[0][1][0, 0]) == (255, 0, 0)


def test_rejects_wrong_frame_size(tmp_path: Path) -> None:
    path = _write_avi(tmp_path / "a.avi", [_image()])
    raw = bytearray(path.read_bytes())
    at = raw.find(b"00db")
    raw[at + 4:at + 8] = struct.pack("<I", 2)
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="bytes"):
        list(dib_frames(path))
