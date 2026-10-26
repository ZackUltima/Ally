"""The licence register (`sources.toml`) and the check every dialogue passes (T49, D-19)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from ally_companion.dialogue import VARIANTS, Dialogue, Variant


@dataclass(frozen=True)
class SourceEntry:
    name: str
    licence: str
    url: str
    variants: frozenset[Variant]
    synthetic: bool


def load_register(path: Path) -> dict[str, SourceEntry]:
    with path.open("rb") as f:
        data = tomllib.load(f)
    register: dict[str, SourceEntry] = {}
    for name, raw in data.items():
        entry = cast(dict[str, object], raw)
        variants: set[Variant] = set()
        for v in cast(list[object], entry["variants"]):
            match = next((a for a in VARIANTS if a == v), None)
            if match is None:
                raise ValueError(f"{path}: source {name}: unknown variant {v!r}")
            variants.add(match)
        register[name] = SourceEntry(
            name=name,
            licence=str(entry["licence"]),
            url=str(entry["url"]),
            variants=frozenset(variants),
            synthetic=bool(entry["synthetic"]),
        )
    return register


def check_against_register(d: Dialogue, register: dict[str, SourceEntry]) -> list[str]:
    """Problems with one dialogue's source, licence, variants and synthetic flag; empty if fine."""
    entry = register.get(d.source)
    if entry is None:
        return [f"{d.id}: source {d.source!r} is not in the licence register"]
    problems: list[str] = []
    if d.licence != entry.licence:
        problems.append(
            f"{d.id}: licence {d.licence!r} differs from the register's {entry.licence!r}"
        )
    for v in sorted(d.variants - entry.variants):
        problems.append(f"{d.id}: source {d.source!r} is not allowed in variant {v!r}")
    if d.synthetic != entry.synthetic:
        problems.append(
            f"{d.id}: synthetic={d.synthetic} differs from the register's {entry.synthetic}"
        )
    return problems
