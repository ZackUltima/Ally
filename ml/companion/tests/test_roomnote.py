import json
from dataclasses import replace
from pathlib import Path

import pytest

from ally_companion.roomnote import (
    NOTE_KEYS,
    RoomNote,
    SeenObject,
    build_note,
    note_from_json,
    parse_note,
    people_of,
)

VECTORS = json.loads(
    (Path(__file__).parent / "fixtures" / "roomnote_vectors.json").read_text(encoding="utf-8")
)

EXAMPLE = RoomNote(
    time="Friday 23 October, 7:40 pm",
    people="one",
    objects=(SeenObject("cup", "table", small=False), SeenObject("glasses", "sofa", small=True)),
    talk="comforting",
    notes=("granddaughter Mei visits on Sundays", "likes gardening"),
)
EXAMPLE_TEXT = (
    "<room>\n"
    "time: Friday 23 October, 7:40 pm\n"
    "people: one\n"
    "objects: cup (table); glasses (sofa, small)\n"
    "talk: comforting\n"
    "notes: granddaughter Mei visits on Sundays; likes gardening\n"
    "</room>"
)


def test_the_example_note_is_built_exactly() -> None:  # D-66 decision 2, byte for byte
    assert build_note(EXAMPLE) == EXAMPLE_TEXT


def test_keys_come_in_a_fixed_order() -> None:
    assert NOTE_KEYS == ("time", "people", "objects", "talk", "notes")


def test_empty_fields_are_left_out() -> None:
    text = build_note(RoomNote("Friday 23 October, 7:40 pm", "one"))
    assert "talk:" not in text and "objects:" not in text and "notes:" not in text


def test_notes_are_left_out_unless_one_person() -> None:  # design choice 46
    assert "notes:" not in build_note(replace(EXAMPLE, people="more"))
    assert "notes:" not in build_note(replace(EXAMPLE, people="none"))


def test_people_says_more_for_two_or_more() -> None:  # never a count (D-33)
    assert [people_of(n) for n in (0, 1, 2, 5)] == ["none", "one", "more", "more"]
    with pytest.raises(ValueError):
        people_of(-1)


def test_build_and_parse_round_trip() -> None:  # the phone's Kotlin builder must pass these (CM3)
    assert len(VECTORS) == 5
    for v in VECTORS:
        note = note_from_json(v["note"])
        assert build_note(note) == v["text"], v["name"]
        assert parse_note(v["text"]) == note, v["name"]


def test_a_semicolon_in_an_object_name_is_refused() -> None:
    with pytest.raises(ValueError, match="object"):
        build_note(RoomNote("now", "one", objects=(SeenObject("cup; plate", None, small=False),)))


def test_marks_that_would_break_the_format_are_refused() -> None:
    for bad in ("sofa, left", "box (blue)", "two\nlines"):
        with pytest.raises(ValueError):
            build_note(RoomNote("now", "one", objects=(SeenObject("cup", bad, small=False),)))
    with pytest.raises(ValueError, match="note"):
        build_note(RoomNote("now", "one", notes=("likes tea; and cake",)))
    with pytest.raises(ValueError):
        build_note(RoomNote("now\nlater", "one"))


def test_parse_refuses_text_that_is_not_a_note() -> None:
    with pytest.raises(ValueError):
        parse_note("time: now\npeople: one")
    with pytest.raises(ValueError):
        parse_note("<room>\npeople: one\ntime: now\n</room>")  # keys out of order
    with pytest.raises(ValueError):
        parse_note("<room>\ntime: now\npeople: two\n</room>")
