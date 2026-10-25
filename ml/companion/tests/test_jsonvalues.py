import pytest

from ally_companion.jsonvalues import as_bool, as_list, as_map, as_str, one_of


def test_values_of_the_wrong_kind_are_refused() -> None:
    for check, bad in ((as_str, 3), (as_bool, "yes"), (as_list, {}), (as_map, [])):
        with pytest.raises(ValueError):
            check(bad)
    with pytest.raises(ValueError, match="colour"):
        one_of("green", ("red", "blue"), "colour")


def test_values_of_the_right_kind_pass_through() -> None:
    assert (as_str("a"), as_bool(False)) == ("a", False)
    assert (as_list([1]), as_map({"k": 1})) == ([1], {"k": 1})
    assert one_of("blue", ("red", "blue"), "colour") == "blue"
