import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from ally_emotion.crop import Box, area_resize, crop_face, square_box, to_grey

VECTORS = Path(__file__).parent / "fixtures" / "crop_vectors.json"


def solid(h: int, w: int, rgb: tuple[int, int, int]) -> NDArray[np.uint8]:
    return np.broadcast_to(np.array(rgb, dtype=np.uint8), (h, w, 3)).copy()


def rgb(h: int, w: int) -> NDArray[np.uint8]:
    ys, xs = np.mgrid[0:h, 0:w]
    return np.stack([xs * 2 % 256, ys * 3 % 256, (xs + ys) % 256], axis=-1).astype(np.uint8)


def test_output_is_48_by_48_one_channel() -> None:
    out = crop_face(rgb(100, 120), Box(10, 10, 60, 80), 0.2)
    assert out.shape == (48, 48) and out.dtype == np.uint8


def test_a_pure_red_pixel_is_76() -> None:  # floor(0.299 * 255 + 0.5)
    assert crop_face(solid(96, 96, (255, 0, 0)), Box(0, 0, 96, 96), 0.0)[0, 0] == 76


def test_grey_weights_are_bt601() -> None:
    primaries = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    assert to_grey(primaries).tolist() == [[0.299 * 255, 0.587 * 255, 0.114 * 255]]


def test_square_box_is_centred_and_enlarged() -> None:
    assert square_box(Box(10, 20, 40, 60), 0.25) == Box(-7.5, 12.5, 75, 75)


def test_two_by_two_blocks_average_exactly() -> None:
    grey = np.zeros((96, 96), dtype=np.uint8)
    grey[0::2, 0::2], grey[0::2, 1::2] = 10, 11
    grey[1::2, 0::2], grey[1::2, 1::2] = 12, 13  # each 2 x 2 block: mean 11.5
    out = crop_face(grey, Box(0, 0, 96, 96), 0.0)
    assert (out == 12).all()  # floor(11.5 + 0.5)


def test_area_resize_of_a_square_array() -> None:
    grey = np.arange(16, dtype=np.float64).reshape(4, 4)
    assert area_resize(grey, 2).tolist() == [[2.5, 4.5], [10.5, 12.5]]
    with pytest.raises(ValueError):
        area_resize(np.zeros((4, 6)), 2)


def test_a_box_outside_the_picture_is_black_there() -> None:  # Review Focus 1
    out = crop_face(solid(96, 96, (200, 200, 200)), Box(-48, 0, 96, 96), 0.0)
    assert (out[:, :24] == 0).all() and (out[:, 24:] == 200).all()
    # past the right and bottom edges
    edge = crop_face(solid(96, 96, (200, 200, 200)), Box(60, 70, 50, 50), 0.3)
    assert edge.shape == (48, 48) and edge[0, 0] == 200 and edge[-1, -1] == 0


def test_a_box_wholly_outside_is_all_black() -> None:
    assert (crop_face(solid(50, 50, (255, 255, 255)), Box(500, -300, 40, 40), 0.1) == 0).all()


def test_grey_input_is_accepted() -> None:
    grey = np.full((60, 60), 99, dtype=np.uint8)
    assert (crop_face(grey, Box(0, 0, 60, 60), 0.0) == 99).all()


def test_bad_input_is_refused() -> None:
    with pytest.raises(ValueError):
        crop_face(np.zeros((10, 10, 4), dtype=np.uint8), Box(0, 0, 10, 10), 0.0)
    with pytest.raises(ValueError):
        crop_face(np.zeros((10, 10), dtype=np.float64), Box(0, 0, 10, 10), 0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        crop_face(np.zeros((10, 10), dtype=np.uint8), Box(0, 0, 0, 0), 0.0)


def test_vectors_file_matches() -> None:  # the Kotlin crop must reproduce these too (T14, EM3)
    cases = json.loads(VECTORS.read_text(encoding="utf-8"))
    assert len(cases) == 5
    for case in cases:
        image = np.array(case["image"], dtype=np.uint8)
        box = Box(*case["box"])
        out = crop_face(image, box, case["margin"])
        assert out.tolist() == case["expected"], case["name"]
