"""Every recorded session must replay to its expected event sequence (ADR-004, evaluation.md "Harness")."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SESSIONS = sorted(p.parent for p in HERE.glob("*/expected_events.json"))


def _expected(session: Path) -> list[dict]:
    data = json.loads((session / "expected_events.json").read_text(encoding="utf-8"))
    assert isinstance(data, list), "expected_events.json must be an ordered list"
    for row in data:
        assert {"type", "t_min", "t_max"} <= row.keys(), f"{session.name}: bad row {row}"
        assert row["t_min"] <= row["t_max"]
    return data


@pytest.mark.skipif(not SESSIONS, reason="no replay sessions recorded yet (first ones arrive in Sprint 1)")
@pytest.mark.parametrize("session", SESSIONS, ids=[s.name for s in SESSIONS])
def test_session_replays_to_expected_events(session: Path):
    expected = _expected(session)
    pytest.importorskip("ally.replay", reason="replay runner (ally/replay.py) lands in Sprint 1")
    from ally.replay import replay_session  # type: ignore[import-not-found]

    emitted = replay_session(session)  # list[Event]
    assert [e.type.value for e in emitted] == [row["type"] for row in expected]
    for ev, row in zip(emitted, expected, strict=True):
        assert row["t_min"] <= ev.ts <= row["t_max"], f"{ev.type} at {ev.ts:.2f}s outside the expected window"
