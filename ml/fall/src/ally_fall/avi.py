"""Frames of uncompressed (24-bit DIB) AVI files, read without OpenCV.

OpenCV 5.0.0 corrupts memory when it converts these frames: on Le2i (2026-10-20), grab()
alone never crashed, while retrieve() or read() crashed 8 of 8 runs. video_frames() sends
such files here. Frame i is at i * scale / rate seconds (the stream header's timing).
"""

from __future__ import annotations

import mmap
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_BI_RGB = 0


@dataclass(frozen=True)
class _Video:
    stream: int
    width: int
    height: int  # positive: rows stored bottom-up
    bits: int
    compression: int
    scale: int
    rate: int


def _chunks(data: bytes | mmap.mmap, start: int, end: int) -> Iterator[tuple[bytes, int, int]]:
    """(fourcc, data offset, size) for each chunk between start and end."""
    pos = start
    while pos + 8 <= end:
        fourcc = bytes(data[pos:pos + 4])
        (size,) = struct.unpack("<I", data[pos + 4:pos + 8])
        yield fourcc, pos + 8, size
        pos += 8 + size + (size & 1)


def _lists(data: bytes | mmap.mmap, start: int, end: int, kind: bytes) -> Iterator[tuple[int, int]]:
    for fourcc, offset, size in _chunks(data, start, end):
        if fourcc == b"LIST" and bytes(data[offset:offset + 4]) == kind:
            yield offset + 4, offset + size


def _video(data: bytes | mmap.mmap) -> _Video | None:
    """The video stream's format, or None if this is not a RIFF AVI with a video stream."""
    if len(data) < 12 or bytes(data[0:4]) != b"RIFF" or bytes(data[8:12]) != b"AVI ":
        return None
    (riff_size,) = struct.unpack("<I", data[4:8])
    end = min(len(data), 8 + riff_size)
    for hdrl_start, hdrl_end in _lists(data, 12, end, b"hdrl"):
        streams = _lists(data, hdrl_start, hdrl_end, b"strl")
        for stream, (strl_start, strl_end) in enumerate(streams):
            parts = {fourcc: (offset, size) for fourcc, offset, size in
                     _chunks(data, strl_start, strl_end)}
            strh, strf = parts.get(b"strh"), parts.get(b"strf")
            if strh is None or strf is None or bytes(data[strh[0]:strh[0] + 4]) != b"vids":
                continue
            scale, rate = struct.unpack("<II", data[strh[0] + 20:strh[0] + 28])
            _, width, height, _, bits, compression = struct.unpack(
                "<IiiHHI", data[strf[0]:strf[0] + 20]
            )
            if b"indx" in parts:
                raise ValueError("OpenDML AVI (indx) is not supported")
            return _Video(stream, width, height, bits, compression, scale, rate)
    return None


def is_uncompressed_avi(path: Path) -> bool:
    """True if the file is an AVI whose video stream is stored uncompressed (BI_RGB)."""
    with open(path, "rb") as file:
        head = file.read(65536)
    try:
        video = _video(head)
    except (ValueError, struct.error):
        return False
    return video is not None and video.compression == _BI_RGB


def dib_frames(path: Path) -> Iterator[tuple[float, NDArray[np.uint8]]]:
    """(time in seconds, upright RGB image) for every frame of a 24-bit uncompressed AVI."""
    with open(path, "rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as data:
        video = _video(data)
        if video is None:
            raise ValueError(f"{path}: not an AVI with a video stream")
        if video.compression != _BI_RGB:
            raise ValueError(f"{path}: video is not uncompressed (compression {video.compression})")
        if video.bits != 24:
            raise ValueError(f"{path}: only 24-bit frames are supported, not {video.bits}-bit")
        if video.scale <= 0 or video.rate <= 0:
            raise ValueError(
                f"{path}: no usable frame rate (scale {video.scale}, rate {video.rate})"
            )
        (riff_size,) = struct.unpack("<I", data[4:8])
        riff_end = 8 + riff_size + (riff_size & 1)
        if len(data) >= riff_end + 12 and data[riff_end:riff_end + 4] == b"RIFF" \
                and data[riff_end + 8:riff_end + 12] == b"AVIX":
            raise ValueError(f"{path}: OpenDML AVI (AVIX) is not supported")

        width, height = video.width, abs(video.height)
        row = ((width * 24 + 31) // 32) * 4
        frame_bytes = row * height
        ids = {f"{video.stream:02d}db".encode(), f"{video.stream:02d}dc".encode()}

        def frames(start: int, end: int) -> Iterator[tuple[int, int]]:
            for fourcc, offset, size in _chunks(data, start, end):
                if fourcc == b"LIST" and data[offset:offset + 4] == b"rec ":
                    yield from frames(offset + 4, offset + size)
                elif fourcc in ids:
                    yield offset, size

        index = 0
        for movi_start, movi_end in _lists(data, 12, min(len(data), 8 + riff_size), b"movi"):
            for offset, size in frames(movi_start, movi_end):
                if size != frame_bytes:
                    raise ValueError(
                        f"{path}: frame {index} has {size} bytes, expected {frame_bytes}"
                    )
                rows = np.frombuffer(data[offset:offset + size], dtype=np.uint8)  # a copy
                bgr = rows.reshape(height, row)[:, : width * 3].reshape(height, width, 3)
                if video.height > 0:
                    bgr = bgr[::-1]
                yield index * video.scale / video.rate, np.ascontiguousarray(bgr[..., ::-1])
                index += 1
