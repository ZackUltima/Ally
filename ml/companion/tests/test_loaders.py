import json
from collections.abc import Mapping
from pathlib import Path

from ally_companion.loaders import LoadReport, empathetic_dialogues, sample, soda_dialogues
from ally_companion.sources import check_against_register, load_register

FIXTURES = Path(__file__).parent / "fixtures"
REGISTER = load_register(Path(__file__).resolve().parents[1] / "sources.toml")


def _rows(name: str) -> list[Mapping[str, object]]:
    lines = (FIXTURES / name).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_soda_maps_two_speakers() -> None:
    first = next(soda_dialogues(_rows("soda_sample.jsonl")))
    assert first.id == "soda-0"
    assert [(t.role, t.text) for t in first.turns] == [
        ("person", "Line A1."),
        ("ally", "Line B1."),
        ("person", "Line A2."),
        ("ally", "Line B2."),
    ]
    assert first.synthetic and first.variants == frozenset({"fyp", "clean"})


def test_soda_three_speakers_are_skipped_and_counted() -> None:  # Review Focus 3
    report = LoadReport()
    ids = [d.id for d in soda_dialogues(_rows("soda_sample.jsonl"), report)]
    assert "soda-1" not in ids
    assert report.skipped["three or more speakers"] == 1


def test_a_dialogue_ending_on_the_person_drops_its_last_turn() -> None:
    by_id = {d.id: d for d in soda_dialogues(_rows("soda_sample.jsonl"))}
    assert [t.text for t in by_id["soda-2"].turns] == ["Line A1.", "Line B1."]


def test_turns_that_do_not_alternate_or_are_too_short_are_skipped_and_counted() -> None:
    report = LoadReport()
    ids = [d.id for d in soda_dialogues(_rows("soda_sample.jsonl"), report)]
    assert ids == ["soda-0", "soda-2"]
    assert report.kept == 2
    assert report.skipped == {
        "three or more speakers": 1,
        "turns do not alternate": 1,
        "too short": 1,
    }


def test_ed_groups_and_orders_utterances() -> None:
    first = next(empathetic_dialogues(_rows("ed_sample.jsonl")))
    assert first.id == "ed-hit:1_conv:2"
    assert [t.role for t in first.turns] == ["person", "ally"]  # the third turn ends on the person
    assert first.turns[0].text == "Line A1."


def test_ed_comma_placeholder_is_restored() -> None:  # Review Focus 4
    text = next(empathetic_dialogues(_rows("ed_sample.jsonl"))).turns[1].text
    assert "," in text and "_comma_" not in text
    assert text == "Line B1, second part."


def test_ed_is_fyp_only_and_not_synthetic() -> None:
    report = LoadReport()
    loaded = list(empathetic_dialogues(_rows("ed_sample.jsonl"), report))
    assert all(not d.synthetic and d.variants == frozenset({"fyp"}) for d in loaded)
    assert [d.id for d in loaded] == ["ed-hit:1_conv:2", "ed-hit:3_conv:6"]
    assert report.skipped == {"turns do not alternate": 1}


def test_loaded_dialogues_pass_the_licence_register() -> None:
    loaded = [
        *soda_dialogues(_rows("soda_sample.jsonl")),
        *empathetic_dialogues(_rows("ed_sample.jsonl")),
    ]
    assert all(check_against_register(d, REGISTER) == [] for d in loaded)


def test_sampling_is_repeatable() -> None:
    loaded = list(soda_dialogues(_rows("soda_sample.jsonl"))) * 5
    assert sample(loaded, 3, seed=0) == sample(loaded, 3, seed=0)
    assert len(sample(loaded, 3, seed=0)) == 3
    assert len(sample(loaded, 100, seed=0)) == len(loaded)
