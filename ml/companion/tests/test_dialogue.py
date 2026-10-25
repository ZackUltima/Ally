from pathlib import Path

import pytest

from ally_companion.dialogue import Dialogue, Turn, read_jsonl, write_jsonl
from ally_companion.roomnote import RoomNote, SeenObject

NOTE = RoomNote(
    time="Friday 23 October, 7:40 pm",
    people="one",
    objects=(SeenObject("cup", "table", small=False),),
    talk="comforting",
    notes=("likes gardening",),
)


def _dialogue(turns: tuple[Turn, ...]) -> Dialogue:
    return Dialogue(
        id="made-up-1",
        source="soda",
        licence="CC BY 4.0",
        synthetic=True,
        variants=frozenset({"fyp", "clean"}),
        turns=turns,
    )


def test_a_dialogue_round_trips_through_jsonl(tmp_path: Path) -> None:
    d = _dialogue((Turn("person", "Made-up words.", NOTE), Turn("ally", "Made-up reply.")))
    path = tmp_path / "d.jsonl"
    assert write_jsonl(path, [d, d]) == 2
    assert list(read_jsonl(path)) == [d, d]


def test_turns_must_alternate_and_end_with_ally() -> None:
    with pytest.raises(ValueError, match="alternate"):
        _dialogue((Turn("person", "a"), Turn("person", "b"), Turn("ally", "c")))
    with pytest.raises(ValueError, match="end"):
        _dialogue((Turn("person", "a"), Turn("ally", "b"), Turn("person", "c")))
    with pytest.raises(ValueError):
        _dialogue(())


def test_a_bad_role_in_a_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    write_jsonl(path, [_dialogue((Turn("person", "a"), Turn("ally", "b")))])
    path.write_text(path.read_text(encoding="utf-8").replace('"ally"', '"robot"'), encoding="utf-8")
    with pytest.raises(ValueError, match="role"):
        list(read_jsonl(path))
