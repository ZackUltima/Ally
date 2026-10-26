import json
from datetime import date
from pathlib import Path

import pytest

from ally_companion.dialogue import Dialogue, Turn, write_jsonl
from ally_companion.heldout import (
    HELDOUT_FILES,
    HeldoutChanged,
    NoManifest,
    filter_training,
    near_duplicates,
    normalise,
    verify_manifest,
    write_manifest,
)

# Placeholder words only: the real held-out sets are written in CM1b, never by Claude (D-66).
SCENARIO = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"  # 6 five-word runs


def _d(dialogue_id: str, *texts: str) -> Dialogue:
    turns = tuple(Turn("person" if i % 2 == 0 else "ally", t) for i, t in enumerate(texts))
    return Dialogue(dialogue_id, "soda", "CC BY 4.0", True, frozenset({"fyp", "clean"}), turns)


def _heldout(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    write_jsonl(folder / "scenarios.jsonl", [_d("s-1", SCENARIO, "reply placeholder")])
    (folder / "crisis.jsonl").write_text(json.dumps({"text": "kilo lima mike"}) + "\n", "utf-8")
    (folder / "ordinary.jsonl").write_text(json.dumps({"text": "help me"}) + "\n", "utf-8")


def test_normalise_lowers_and_drops_punctuation() -> None:
    assert normalise("Hello, THERE!  It's ok…") == ["hello", "there", "its", "ok"]


def test_no_manifest_no_training_data(tmp_path: Path) -> None:  # T48
    _heldout(tmp_path / "heldout")
    with pytest.raises(NoManifest):
        filter_training([_d("t-1", "a", "b")], tmp_path / "heldout", tmp_path / "heldout.md")


def test_the_manifest_holds_hashes_and_counts_not_text(tmp_path: Path) -> None:
    _heldout(tmp_path / "heldout")
    hashes = write_manifest(tmp_path / "heldout", tmp_path / "heldout.md", today=date(2026, 11, 2))
    assert set(hashes) == set(HELDOUT_FILES) and all(len(h) == 64 for h in hashes.values())
    report = (tmp_path / "heldout.md").read_text(encoding="utf-8")
    assert "2026-11-02" in report and all(h in report for h in hashes.values())
    assert "alpha" not in report and "kilo" not in report
    verify_manifest(tmp_path / "heldout", tmp_path / "heldout.md")  # unchanged: no error


def test_a_missing_heldout_file_is_refused(tmp_path: Path) -> None:
    _heldout(tmp_path / "heldout")
    (tmp_path / "heldout" / "ordinary.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="ordinary"):
        write_manifest(tmp_path / "heldout", tmp_path / "heldout.md")


def test_an_edited_heldout_file_stops_the_build(tmp_path: Path) -> None:  # Review Focus 5
    _heldout(tmp_path / "heldout")
    write_manifest(tmp_path / "heldout", tmp_path / "heldout.md")
    with (tmp_path / "heldout" / "crisis.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"text": "november oscar papa"}) + "\n")
    with pytest.raises(HeldoutChanged, match="crisis.jsonl"):
        filter_training([_d("t-1", "a", "b")], tmp_path / "heldout", tmp_path / "heldout.md")


def test_a_copied_scenario_is_removed() -> None:  # Jaccard 1.0
    scenarios = [_d("s-1", SCENARIO, "x")]
    found = near_duplicates([_d("t-1", SCENARIO.upper() + "!", "y")], scenarios, [])
    assert found == {"t-1": "scenario overlap"}


def test_a_reworded_scenario_below_the_threshold_stays() -> None:
    scenarios = [_d("s-1", SCENARIO, "x")]
    # 2 runs shared, 10 in all: Jaccard 0.2
    reworded = "alpha bravo charlie delta echo foxtrot xray yankee zulu whiskey"
    assert near_duplicates([_d("t-1", reworded, "y")], scenarios, []) == {}


def test_a_crisis_phrase_inside_a_dialogue_is_removed() -> None:  # 3+ words, word for word
    training = [
        _d("t-1", "one two", "three four"),
        _d("t-2", "and then KILO, lima mike again", "z"),
    ]
    assert near_duplicates(training, [], ["kilo lima mike"]) == {"t-2": "heldout phrase"}


def test_a_short_phrase_counts_only_as_a_whole_turn() -> None:
    training = [_d("t-1", "can you help me find my glasses", "y"), _d("t-2", "Help me!", "y")]
    assert near_duplicates(training, [], ["help me"]) == {"t-2": "heldout phrase"}


def test_the_report_counts_each_rule(tmp_path: Path) -> None:
    _heldout(tmp_path / "heldout")
    write_manifest(tmp_path / "heldout", tmp_path / "heldout.md")
    training = [
        _d("t-1", SCENARIO, "y"),
        _d("t-2", "x kilo lima mike", "y"),
        _d("t-3", "help me", "y"),
        _d("t-4", "an ordinary made up line", "y"),
    ]
    kept, removed = filter_training(training, tmp_path / "heldout", tmp_path / "heldout.md")
    assert [d.id for d in kept] == ["t-4"]
    assert removed == {"scenario overlap": 1, "heldout phrase": 2}
