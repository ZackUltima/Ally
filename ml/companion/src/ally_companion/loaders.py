"""SODA and EmpatheticDialogues rows into the one dialogue form (technical plan section C2, T49).

The first speaker is the person and the other is Ally. A dialogue with three or more speakers,
or turns that do not alternate, is skipped and counted, never mapped by turn order; a last person
turn is dropped so the dialogue ends on an Ally target. Licence values match `sources.toml`.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field

from ally_companion.dialogue import Dialogue, Role, Turn, Variant
from ally_companion.jsonvalues import as_list, as_str

SODA_LICENCE = "CC BY 4.0"
ED_LICENCE = "CC BY-NC 4.0"
_COMMA = "_comma_"


@dataclass
class LoadReport:
    kept: int = 0
    skipped: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1


def soda_dialogues(
    rows: Iterable[Mapping[str, object]], report: LoadReport | None = None
) -> Iterator[Dialogue]:
    """SODA rows (`dialogue`, `speakers`, `original_index`): machine-generated, both variants."""
    counts = report if report is not None else LoadReport()
    for number, row in enumerate(rows):
        index = row.get("original_index", number)
        speakers = [as_str(s) for s in as_list(row["speakers"])]
        texts = [as_str(t) for t in as_list(row["dialogue"])]
        d = _dialogue(
            f"soda-{index}",
            "soda",
            SODA_LICENCE,
            True,
            frozenset({"fyp", "clean"}),
            speakers,
            texts,
            counts,
        )
        if d is not None:
            yield d


def empathetic_dialogues(
    rows: Iterable[Mapping[str, object]], report: LoadReport | None = None
) -> Iterator[Dialogue]:
    """EmpatheticDialogues rows, grouped by `conv_id` and ordered by `utterance_idx`:
    crowdsourced, FYP variant only (D-17 (a))."""
    counts = report if report is not None else LoadReport()
    conversations: dict[str, list[tuple[int, str, str]]] = {}
    for row in rows:
        utterance = as_str(row["utterance"]).replace(_COMMA, ",").strip()
        conversations.setdefault(as_str(row["conv_id"]), []).append(
            (_as_int(row["utterance_idx"]), str(_as_int(row["speaker_idx"])), utterance)
        )
    for conv_id, utterances in conversations.items():
        ordered = sorted(utterances)
        d = _dialogue(
            f"ed-{conv_id}",
            "empathetic_dialogues",
            ED_LICENCE,
            False,
            frozenset({"fyp"}),
            [speaker for _, speaker, _ in ordered],
            [text for _, _, text in ordered],
            counts,
        )
        if d is not None:
            yield d


def sample(dialogues: Iterable[Dialogue], n: int, seed: int) -> list[Dialogue]:
    """A repeatable random sample of at most n dialogues."""
    pool = list(dialogues)
    return random.Random(seed).sample(pool, min(n, len(pool)))


def _dialogue(
    dialogue_id: str,
    source: str,
    licence: str,
    synthetic: bool,
    variants: frozenset[Variant],
    speakers: Sequence[str],
    texts: Sequence[str],
    report: LoadReport,
) -> Dialogue | None:
    if len(speakers) != len(texts):
        raise ValueError(f"{dialogue_id}: {len(speakers)} speakers for {len(texts)} turns")
    if len(set(speakers)) > 2:
        report.skip("three or more speakers")
        return None
    if any(a == b for a, b in zip(speakers, speakers[1:], strict=False)):
        report.skip("turns do not alternate")
        return None
    roles: list[Role] = ["person" if s == speakers[0] else "ally" for s in speakers]
    turns = [Turn(role, text) for role, text in zip(roles, texts, strict=True)]
    if turns and turns[-1].role == "person":
        turns = turns[:-1]
    if len(turns) < 2:
        report.skip("too short")
        return None
    report.kept += 1
    return Dialogue(dialogue_id, source, licence, synthetic, variants, tuple(turns))


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"expected a whole number, got {value!r}")
    return value
