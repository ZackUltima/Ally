"""Writes crop_vectors.json: 5 made-up images and boxes with the 48 x 48 pixels crop_face gives.

The Kotlin crop (sprint EM3) must reproduce every expected pixel (T14). Run from ml/emotion/:
    PYTHONPATH=src python tests/fixtures/make_crop_vectors.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ally_emotion.crop import Box, crop_face


def gradient(h: int, w: int) -> NDArray[np.uint8]:
    ys, xs = np.mgrid[0:h, 0:w]
    return ((xs * 255 // max(w - 1, 1) + ys * 3) % 256).astype(np.uint8)


def checkerboard(h: int, w: int) -> NDArray[np.uint8]:
    ys, xs = np.mgrid[0:h, 0:w]
    on = ((xs // 5 + ys // 5) % 2).astype(bool)
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[on] = (250, 40, 10)
    image[~on] = (10, 60, 30)  # grey about 99 against about 42
    return image


def colours(h: int, w: int) -> NDArray[np.uint8]:
    ys, xs = np.mgrid[0:h, 0:w]
    return np.stack([xs * 7 % 256, ys * 11 % 256, (xs * ys) % 256], axis=-1).astype(np.uint8)


CASES: list[tuple[str, NDArray[np.uint8], Box, float]] = [
    ("grey gradient, whole image, scale 1.25", gradient(60, 60), Box(0, 0, 60, 60), 0.0),
    ("RGB checkerboard, inner box, margin 0.2", checkerboard(40, 50), Box(5, 5, 30, 30), 0.2),
    ("box past the left and top edges", colours(40, 40), Box(-10, -5, 30, 30), 0.0),
    ("box past the right and bottom edges", colours(40, 40), Box(25, 30, 30, 20), 0.1),
    ("non-integer box and scale", colours(32, 32), Box(3.3, 7.7, 21.1, 17.9), 0.15),
]


def main() -> None:
    out = [
        {
            "name": name,
            "image": image.tolist(),
            "box": [box.x, box.y, box.w, box.h],
            "margin": margin,
            "expected": crop_face(image, box, margin).tolist(),
        }
        for name, image, box, margin in CASES
    ]
    path = Path(__file__).with_name("crop_vectors.json")
    path.write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {len(out)} cases to {path}")


if __name__ == "__main__":
    main()
