"""One typed form for every dialogue, with its source, licence and synthetic flag (T49, D-19).

A dialogue alternates person and Ally turns and ends with an Ally turn, because a training
example ends on the target reply. Files are JSON Lines under ~/ally-data/companion/ (D-52).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from ally_companion.jsonvalues import as_bool, as_list, as_map, as_str, one_of
from ally_companion.roomnote import RoomNote, note_from_json, note_to_json

Role = Literal["person", "ally"]
Variant = Literal["fyp", "clean"]

ROLES: tuple[Role, ...] = ("person", "ally")
VARIANTS: tuple[Variant, ...] = ("fyp", "clean")


@dataclass(frozen=True)
class Turn:
    role: Role
    text: str
    note: RoomNote | None = None


@dataclass(frozen=True)
class Dialogue:
    id: str
    source: str
    licence: str
    synthetic: bool
    variants: frozenset[Variant]
    turns: tuple[Turn, ...]

    def __post_init__(self) -> None:
        if not self.turns:
            raise ValueError(f"dialogue {self.id}: no turns")
        for before, after in zip(self.turns, self.turns[1:], strict=False):
            if before.role == after.role:
                raise ValueError(f"dialogue {self.id}: turns must alternate person and ally")
        if self.turns[-1].role != "ally":
            raise ValueError(f"dialogue {self.id}: must end with an ally turn (the target)")
        if not self.variants:
            raise ValueError(f"dialogue {self.id}: no variant")


def write_jsonl(path: Path, dialogues: Iterable[Dialogue]) -> int:
    """Write one dialogue per line; returns how many were written."""
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for d in dialogues:
            f.write(json.dumps(_to_json(d), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[Dialogue]:
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: not a JSON object")
                yield _from_json(cast(Mapping[str, object], row))


def _to_json(d: Dialogue) -> dict[str, object]:
    return {
        "id": d.id,
        "source": d.source,
        "licence": d.licence,
        "synthetic": d.synthetic,
        "variants": sorted(d.variants),
        "turns": [_turn_to_json(t) for t in d.turns],
    }


def _turn_to_json(t: Turn) -> dict[str, object]:
    out: dict[str, object] = {"role": t.role, "text": t.text}
    if t.note is not None:
        out["note"] = note_to_json(t.note)
    return out


def _from_json(row: Mapping[str, object]) -> Dialogue:
    return Dialogue(
        id=as_str(row["id"]),
        source=as_str(row["source"]),
        licence=as_str(row["licence"]),
        synthetic=as_bool(row["synthetic"]),
        variants=frozenset(one_of(v, VARIANTS, "variant") for v in as_list(row["variants"])),
        turns=tuple(_turn_from_json(as_map(t)) for t in as_list(row["turns"])),
    )


def _turn_from_json(row: Mapping[str, object]) -> Turn:
    note = row.get("note")
    return Turn(
        role=one_of(row["role"], ROLES, "role"),
        text=as_str(row["text"]),
        note=None if note is None else note_from_json(as_map(note)),
    )
