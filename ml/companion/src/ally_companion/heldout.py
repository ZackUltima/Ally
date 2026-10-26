"""The held-out sets are fixed before any training data is built, and kept out of it (T48;
D-66 decisions 3 and 5).

The sets live in ~/ally-data/companion/heldout/ and are never committed; the repository report
holds only their SHA-256 hashes, line counts and the date fixed, so the viva can see they did
not change after training started. `filter_training` refuses to run without that report or when
a file no longer matches it.

Files: scenarios.jsonl (dialogues, as `dialogue.write_jsonl`), crisis.jsonl and ordinary.jsonl
(one {"text": ...} object per line).
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path

from ally_companion.dialogue import Dialogue, read_jsonl
from ally_companion.jsonvalues import as_map, as_str

HELDOUT_FILES: tuple[str, ...] = ("scenarios.jsonl", "crisis.jsonl", "ordinary.jsonl")
SCENARIO_RULE = "scenario overlap"
PHRASE_RULE = "heldout phrase"
MIN_PHRASE_WORDS = 3

_PUNCTUATION = re.compile(r"[^\w\s]")
_ROW = re.compile(r"^\| `(?P<file>[^`]+)` \| (?P<lines>\d+) \| `(?P<sha>[0-9a-f]{64})` \|$")


class NoManifest(FileNotFoundError):
    """The held-out report is missing, so no training data may be built (T48)."""


class HeldoutChanged(RuntimeError):
    """A held-out file no longer matches the hash recorded when the sets were fixed."""


def write_manifest(folder: Path, report: Path, today: date | None = None) -> dict[str, str]:
    """Record each held-out file's SHA-256 and line count, and the date, in `report`."""
    hashes: dict[str, str] = {}
    rows: list[str] = []
    for name in HELDOUT_FILES:
        path = folder / name
        if not path.is_file():
            raise FileNotFoundError(f"held-out file missing: {path}")
        hashes[name] = _sha256(path)
        rows.append(f"| `{name}` | {_line_count(path)} | `{hashes[name]}` |")
    fixed = (today or date.today()).isoformat()
    report.write_text(
        "\n".join(
            [
                "# Companion held-out sets (T48, D-66)",
                "",
                f"Fixed on {fixed}. The sets live in `~/ally-data/companion/heldout/` and "
                "are never committed; this report holds only their hashes and line counts. "
                "Training data is built only while every file still matches "
                "(`ally_companion.heldout`).",
                "",
                "| File | Lines | SHA-256 |",
                "|---|---|---|",
                *rows,
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    return hashes


def verify_manifest(folder: Path, report: Path) -> None:
    if not report.is_file():
        raise NoManifest(f"no held-out report at {report}: fix the held-out sets first (T48)")
    recorded: dict[str, str] = {}
    for line in report.read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line)
        if m:
            recorded[m.group("file")] = m.group("sha")
    for name in HELDOUT_FILES:
        path = folder / name
        if name not in recorded:
            raise HeldoutChanged(f"{name} is not in the held-out report {report}")
        if not path.is_file() or _sha256(path) != recorded[name]:
            raise HeldoutChanged(f"{name} changed after its hash was recorded in {report}")


def normalise(text: str) -> list[str]:
    """NFKC, lower case, punctuation removed, split on spaces."""
    return _PUNCTUATION.sub("", unicodedata.normalize("NFKC", text).lower()).split()


def near_duplicates(
    training: Iterable[Dialogue],
    scenarios: Sequence[Dialogue],
    phrases: Sequence[str],
    jaccard: float = 0.5,
    n: int = 5,
) -> dict[str, str]:
    """Training dialogue id -> the rule that matched (D-66 decision 5).

    Scenario overlap: the person turns' word n-grams overlap one scenario's by Jaccard >= jaccard
    (a turn shorter than n words counts as one gram). Held-out phrase: a phrase of 3+ words
    appears word for word in any turn; a shorter phrase only as a whole turn.
    """
    scenario_grams = [_person_grams(s, n) for s in scenarios]
    index: dict[tuple[str, ...], set[int]] = {}
    for i, grams in enumerate(scenario_grams):
        for g in grams:
            index.setdefault(g, set()).add(i)
    normalised = [tuple(normalise(x)) for x in phrases]
    long_by_length: dict[int, set[tuple[str, ...]]] = {}
    for p in normalised:
        if len(p) >= MIN_PHRASE_WORDS:
            long_by_length.setdefault(len(p), set()).add(p)
    short_phrases = {p for p in normalised if 0 < len(p) < MIN_PHRASE_WORDS}

    found: dict[str, str] = {}
    for d in training:
        grams = _person_grams(d, n)
        candidates = {i for g in grams for i in index.get(g, set())}
        if any(_jaccard(grams, scenario_grams[i]) >= jaccard for i in candidates):
            found[d.id] = SCENARIO_RULE
            continue
        turns = [tuple(normalise(t.text)) for t in d.turns]
        if any(t in short_phrases or _has_phrase(t, long_by_length) for t in turns):
            found[d.id] = PHRASE_RULE
    return found


def filter_training(
    training: Iterable[Dialogue], folder: Path, report: Path
) -> tuple[list[Dialogue], dict[str, int]]:
    """Verify the manifest, then return the kept dialogues and how many each rule removed."""
    verify_manifest(folder, report)
    scenarios = list(read_jsonl(folder / "scenarios.jsonl"))
    phrases = [*_phrases(folder / "crisis.jsonl"), *_phrases(folder / "ordinary.jsonl")]
    pool = list(training)
    found = near_duplicates(pool, scenarios, phrases)
    removed = {SCENARIO_RULE: 0, PHRASE_RULE: 0}
    for rule in found.values():
        removed[rule] += 1
    return [d for d in pool if d.id not in found], removed


def _person_grams(d: Dialogue, n: int) -> set[tuple[str, ...]]:
    grams: set[tuple[str, ...]] = set()
    for t in d.turns:
        if t.role != "person":
            continue
        words = normalise(t.text)
        if 0 < len(words) < n:
            grams.add(tuple(words))
        grams.update(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return grams


def _jaccard(a: set[tuple[str, ...]], b: set[tuple[str, ...]]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def _has_phrase(words: tuple[str, ...], by_length: dict[int, set[tuple[str, ...]]]) -> bool:
    """Set lookups per phrase length, so the cost grows with the words, not with the phrases."""
    return any(
        words[i : i + k] in phrases
        for k, phrases in by_length.items()
        for i in range(len(words) - k + 1)
    )


def _phrases(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [as_str(as_map(json.loads(line))["text"]) for line in lines if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
