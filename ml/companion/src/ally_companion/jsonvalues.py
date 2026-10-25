"""Typed reading of values parsed from JSON: each refuses a value of the wrong kind."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar, cast

T = TypeVar("T", bound=str)


def one_of(value: object, allowed: tuple[T, ...], what: str) -> T:
    for a in allowed:
        if value == a:
            return a
    raise ValueError(f"unknown {what}: {value!r}")


def as_str(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"expected text, got {value!r}")
    return value


def as_bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"expected true or false, got {value!r}")
    return value


def as_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"expected a list, got {value!r}")
    return cast(list[object], value)


def as_map(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"expected an object, got {value!r}")
    return cast(Mapping[str, object], value)
