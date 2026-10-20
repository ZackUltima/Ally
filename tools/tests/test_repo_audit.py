"""Tests for the public-repository audit (D-52)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from repo_audit import find_problems  # noqa: E402


def _no_text(_: str) -> str:
    return ""


SAMPLE_EMAIL = "someone" + "@" + "example.com"

EXCLUDED_PATHS = [
    "research/papers/x.pdf",
    "defense.html.bak-2026-10-15",
    ".mcp.json",
    ".roo/a",
    ".claude/settings.local.json",
    "temporary-for-srd-delete-if-done/a.pdf",
    "ml/fall/data/x.npz",
    "a.mp4",
    "a.avi",
    "a.npz",
    "a.parquet",
    "a.task",
]


def test_flags_excluded_paths() -> None:
    for path in EXCLUDED_PATHS:
        problems = find_problems([path], _no_text)
        assert len(problems) == 1, path
        assert path in problems[0]


def test_allows_ordinary_paths() -> None:
    paths = ["ml/fall/src/ally_fall/ticks.py", "plans/technical-plan.html"]
    assert find_problems(paths, _no_text) == []


def test_flags_personal_details() -> None:
    texts = {
        "a.md": f"contact me at {SAMPLE_EMAIL}",
        "b.html": SAMPLE_EMAIL.replace("@", "&#64;"),
        "c.txt": "SUPABASE_KEY=abc123",
        "d.py": "API_TOKEN = 'xyz'",
    }
    problems = find_problems(list(texts), lambda p: texts[p])
    assert len(problems) == 4
    for path in texts:
        assert any(path in problem for problem in problems)


def test_metric_notation_is_not_an_email() -> None:
    texts = {"a.md": "mAP@0.5 0.979 and mAP@0.5:0.95; PCK@0.2 of 84.1"}
    assert find_problems(list(texts), lambda p: texts[p]) == []


def test_constants_are_not_secrets() -> None:
    texts = {
        "a.py": "MAX_TOKENS = 80  # replies are short",
        "b.py": "SECRET = re.compile(",
    }
    assert find_problems(list(texts), lambda p: texts[p]) == []


def test_public_dataset_contacts_allowed() -> None:
    texts = {"a.html": "e-mail facesdatabase@mpib-berlin.mpg.de or queenie3@live.com"}
    assert find_problems(list(texts), lambda p: texts[p]) == []


def test_ordinary_text_passes() -> None:
    texts = {"a.md": "The fall model reads 30 ticks.", "b.py": "x = 1  # a @decorator"}
    assert find_problems(list(texts), lambda p: texts[p]) == []
