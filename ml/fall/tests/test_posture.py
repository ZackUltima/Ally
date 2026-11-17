"""Tests for the posture groups of the 16 OmniFall labels (P1, D-72)."""

import numpy as np
import pytest

from ally_fall.posture import POSTURES, group_probabilities, posture_map

NAMES = (
    "walk", "fall", "fallen", "sit_down", "sitting", "lie_down", "lying", "stand_up",
    "standing", "other", "kneel_down", "kneeling", "squat_down", "squatting",
    "crawl", "jump",
)


def _posture_of(name: str, names: tuple[str, ...] = NAMES) -> str:
    return POSTURES[int(posture_map(names)[names.index(name)])]


def test_the_five_postures_in_order() -> None:
    assert POSTURES == ("upright", "sitting", "lying", "changing", "other")


@pytest.mark.parametrize(
    ("label", "posture"),
    [
        ("standing", "upright"),
        ("walk", "upright"),
        ("sitting", "sitting"),
        ("lying", "lying"),
        ("lie_down", "lying"),
        ("fallen", "lying"),
        ("fall", "lying"),
        ("sit_down", "changing"),
        ("stand_up", "changing"),
        ("other", "other"),
    ],
)
def test_labels_with_data_map_to_their_posture(label: str, posture: str) -> None:
    assert _posture_of(label) == posture


@pytest.mark.parametrize(
    "label", ["kneel_down", "kneeling", "squat_down", "squatting", "crawl", "jump"]
)
def test_labels_without_training_data_are_other(label: str) -> None:
    assert _posture_of(label) == "other"


def test_the_map_follows_label_names_not_positions() -> None:
    shuffled = tuple(reversed(NAMES))
    assert _posture_of("sitting", shuffled) == "sitting"
    assert _posture_of("fallen", shuffled) == "lying"


def test_every_label_gets_exactly_one_posture() -> None:
    mapped = posture_map(NAMES)
    assert mapped.shape == (16,)
    assert mapped.dtype == np.int64
    assert set(mapped.tolist()) <= set(range(len(POSTURES)))


def test_an_unknown_label_name_raises() -> None:
    with pytest.raises(ValueError, match="upright"):
        posture_map(("fall", "fallen", "lying", "upright", "rest"))


def test_grouped_probabilities_add_up_per_posture_and_still_sum_to_one() -> None:
    probs = np.zeros((2, 16), dtype=np.float32)
    probs[0, NAMES.index("standing")] = 0.5
    probs[0, NAMES.index("walk")] = 0.25
    probs[0, NAMES.index("fallen")] = 0.25
    probs[1, NAMES.index("kneeling")] = 1.0
    grouped = group_probabilities(probs, NAMES)
    assert grouped.shape == (2, 5)
    assert grouped.dtype == np.float32
    assert grouped[0].tolist() == pytest.approx([0.75, 0.0, 0.25, 0.0, 0.0])
    assert grouped[1].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 1.0])
    assert grouped.sum(axis=1) == pytest.approx(np.ones(2))


def test_a_wrong_width_raises() -> None:
    with pytest.raises(ValueError, match="16"):
        group_probabilities(np.zeros((1, 5), dtype=np.float32), NAMES)
