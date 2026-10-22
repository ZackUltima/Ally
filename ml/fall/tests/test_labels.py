from pathlib import Path

import pytest

from ally_fall.labels import GROUPS5, fall_classes, label_map
from ally_fall.omnifall import label_names

NAMES = label_names(Path(__file__).parent / "fixtures" / "omnifall")


def test_groups5_mapping() -> None:
    mapping = label_map(NAMES, "groups5")
    group = {name: GROUPS5[mapping[i]] for i, name in enumerate(NAMES)}
    assert group["fall"] == "fall" and group["fallen"] == "fallen"
    assert group["lie_down"] == group["lying"] == "lying"
    assert group["standing"] == group["walk"] == "upright"
    rest = set(NAMES) - {"fall", "fallen", "lie_down", "lying", "standing", "walk"}
    assert len(rest) == 10 and {group[name] for name in rest} == {"rest"}


def test_omnifall16_is_identity() -> None:
    assert label_map(NAMES, "omnifall16").tolist() == list(range(16))


def test_fall_classes() -> None:
    assert fall_classes("omnifall16", NAMES) == (NAMES.index("fall"), NAMES.index("fallen"))
    assert fall_classes("groups5", NAMES) == (0, 1)


def test_unknown_label_set_rejected() -> None:
    with pytest.raises(ValueError, match="label set"):
        label_map(NAMES, "groups4")  # type: ignore[arg-type]
