"""The room note: what Ally can see and knows, given to the companion model at the start of
each person turn (technical plan section C6, T50; format D-66 decision 2).

It never holds expression readings, images, the mood log, PHQ-2 answers or alert history
(design plan section C7), so the dataclass has no field for them.

    <room>
    time: Friday 23 October, 7:40 pm
    people: one
    objects: cup (table); glasses (sofa, small)
    talk: comforting
    notes: granddaughter Mei visits on Sundays; likes gardening
    </room>

One field per line in NOTE_KEYS order; empty fields are left out; notes only when exactly one
person is in view (design choice 46).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from ally_companion.jsonvalues import as_bool, as_list, as_map, as_str, one_of

People = Literal["none", "one", "more"]
Talk = Literal["check-in", "comforting"]

PEOPLE: tuple[People, ...] = ("none", "one", "more")
TALK: tuple[Talk, ...] = ("check-in", "comforting")
NOTE_KEYS: tuple[str, ...] = ("time", "people", "objects", "talk", "notes")
OPEN, CLOSE = "<room>", "</room>"

_SMALL = "small"
_OBJECT = re.compile(r"^(?P<name>[^();,]+?)(?: \((?P<inside>[^()]+)\))?$")


@dataclass(frozen=True)
class SeenObject:
    name: str
    place: str | None
    small: bool


@dataclass(frozen=True)
class RoomNote:
    time: str
    people: People
    objects: tuple[SeenObject, ...] = ()
    talk: Talk | None = None
    notes: tuple[str, ...] = ()


def people_of(count: int) -> People:
    """How many people are in view, as the note says it: never a count (D-33)."""
    if count < 0:
        raise ValueError(f"a count of people cannot be negative: {count}")
    return PEOPLE[min(count, 2)]


def build_note(note: RoomNote) -> str:
    _plain(note.time, "time", ";")
    lines = [OPEN, f"time: {note.time}", f"people: {note.people}"]
    if note.objects:
        lines.append("objects: " + "; ".join(object_text(o) for o in note.objects))
    if note.talk is not None:
        lines.append(f"talk: {note.talk}")
    if note.notes and note.people == "one":
        for n in note.notes:
            _plain(n, "note", ";")
        lines.append("notes: " + "; ".join(note.notes))
    lines.append(CLOSE)
    return "\n".join(lines)


def parse_note(text: str) -> RoomNote:
    lines = text.split("\n")
    if len(lines) < 4 or lines[0] != OPEN or lines[-1] != CLOSE:
        raise ValueError("a room note starts with <room> and ends with </room>")
    fields: dict[str, str] = {}
    order: list[str] = []
    for line in lines[1:-1]:
        key, sep, value = line.partition(": ")
        if not sep or key not in NOTE_KEYS or key in fields:
            raise ValueError(f"not a room note line: {line!r}")
        fields[key] = value
        order.append(key)
    in_order = order == [k for k in NOTE_KEYS if k in fields]
    if not in_order or "time" not in fields or "people" not in fields:
        raise ValueError("room note keys are missing or out of order")
    talk = fields.get("talk")
    return RoomNote(
        time=fields["time"],
        people=one_of(fields["people"], PEOPLE, "people"),
        objects=tuple(_parse_object(o) for o in fields["objects"].split("; "))
        if "objects" in fields
        else (),
        talk=None if talk is None else one_of(talk, TALK, "talk"),
        notes=tuple(fields["notes"].split("; ")) if "notes" in fields else (),
    )


def note_to_json(note: RoomNote) -> dict[str, object]:
    return {
        "time": note.time,
        "people": note.people,
        "objects": [{"name": o.name, "place": o.place, "small": o.small} for o in note.objects],
        "talk": note.talk,
        "notes": list(note.notes),
    }


def note_from_json(row: Mapping[str, object]) -> RoomNote:
    talk = row.get("talk")
    objects = tuple(
        SeenObject(
            name=as_str(o["name"]),
            place=None if o.get("place") is None else as_str(o["place"]),
            small=as_bool(o["small"]),
        )
        for o in (as_map(x) for x in as_list(row.get("objects", [])))
    )
    return RoomNote(
        time=as_str(row["time"]),
        people=one_of(row["people"], PEOPLE, "people"),
        objects=objects,
        talk=None if talk is None else one_of(talk, TALK, "talk"),
        notes=tuple(as_str(n) for n in as_list(row.get("notes", []))),
    )


def object_text(o: SeenObject) -> str:
    """An object as written: `name`, `name (place)`, `name (small)`, `name (place, small)`."""
    _plain(o.name, "object name", ";(),")
    inside = []
    if o.place is not None:
        _plain(o.place, "object place", ";(),")
        inside.append(o.place)
    if o.small:
        inside.append(_SMALL)
    return f"{o.name} ({', '.join(inside)})" if inside else o.name


def _parse_object(text: str) -> SeenObject:
    m = _OBJECT.match(text)
    if m is None:
        raise ValueError(f"not an object entry: {text!r}")
    parts = m.group("inside").split(", ") if m.group("inside") else []
    small = bool(parts) and parts[-1] == _SMALL
    if small:
        parts = parts[:-1]
    if len(parts) > 1:
        raise ValueError(f"not an object entry: {text!r}")
    return SeenObject(m.group("name"), parts[0] if parts else None, small)


def _plain(value: str, what: str, forbidden: str) -> None:
    """Refuse marks that would break the one-line, semicolon-separated format."""
    if not value.strip() or "\n" in value or any(c in value for c in forbidden):
        raise ValueError(f"{what} {value!r} is empty or holds a newline or one of {forbidden!r}")
