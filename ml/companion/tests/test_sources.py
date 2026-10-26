from dataclasses import replace
from pathlib import Path

import pytest

from ally_companion.dialogue import Dialogue, Turn
from ally_companion.sources import check_against_register, load_register

REGISTER = load_register(Path(__file__).resolve().parents[1] / "sources.toml")

SODA = Dialogue(
    id="soda-1",
    source="soda",
    licence="CC BY 4.0",
    synthetic=True,
    variants=frozenset({"fyp", "clean"}),
    turns=(Turn("person", "a"), Turn("ally", "b")),
)
ED = Dialogue(
    id="ed-1",
    source="empathetic_dialogues",
    licence="CC BY-NC 4.0",
    synthetic=False,
    variants=frozenset({"fyp"}),
    turns=(Turn("person", "a"), Turn("ally", "b")),
)


def test_the_register_holds_soda_and_empathetic_dialogues() -> None:
    assert set(REGISTER) == {"soda", "empathetic_dialogues"}
    assert REGISTER["empathetic_dialogues"].variants == frozenset({"fyp"})


def test_well_formed_dialogues_have_no_problems() -> None:
    assert check_against_register(SODA, REGISTER) == []
    assert check_against_register(ED, REGISTER) == []


def test_an_unknown_source_is_a_problem() -> None:
    problems = check_against_register(replace(SODA, source="somewhere"), REGISTER)
    assert problems and "somewhere" in problems[0]


def test_empathetic_dialogues_in_clean_is_a_problem() -> None:  # D-17 (a)
    problems = check_against_register(replace(ED, variants=frozenset({"fyp", "clean"})), REGISTER)
    assert any("clean" in p for p in problems)


def test_a_synthetic_flag_that_disagrees_with_the_register_is_a_problem() -> None:  # T49
    problems = check_against_register(replace(SODA, synthetic=False), REGISTER)
    assert any("synthetic" in p for p in problems)


def test_a_licence_that_disagrees_with_the_register_is_a_problem() -> None:
    assert any("licence" in p for p in check_against_register(replace(ED, licence="MIT"), REGISTER))


def test_a_register_entry_with_an_unknown_variant_is_refused(tmp_path: Path) -> None:
    bad = tmp_path / "sources.toml"
    bad.write_text(
        '[x]\nlicence = "MIT"\nurl = "https://example.org"\n'
        'variants = ["paid"]\nsynthetic = true\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="variant"):
        load_register(bad)
