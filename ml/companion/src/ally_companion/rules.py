"""Checks of a target reply against the reply rules (design plan section C6; D-66 decision 4).

Code decides only what it can count: 1 to 3 sentences, at most one question, no room-note tag,
key line or exact note entry, no 5 or more words copied from a memory note, and a memory
proposal (<note>...</note>) only at the very end. Advice, secrecy, news and claims to be human
need judgement, so they only raise flags for the curator and never fail a reply.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ally_companion.roomnote import CLOSE, NOTE_KEYS, OPEN, RoomNote, object_text

MAX_SENTENCES = 3
MAX_QUESTIONS = 1
COPY_WORDS = 5

_ABBREVIATIONS = frozenset({"mr.", "mrs.", "ms.", "dr.", "st.", "e.g.", "i.e.", "etc.", "vs."})
_SENTENCE_END = re.compile(r"[.!?]+(?=\s|$)")
_TRAILING_NOTE = re.compile(r"\s*<note>[^<]*</note>\s*$")
_KEY_LINE = re.compile(r"\b(?:" + "|".join(NOTE_KEYS) + r"):\s")
_WORD = re.compile(r"[a-z0-9']+")

_FLAGS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("medical advice", re.compile(r"\b(?:dose|mg|diagnos\w*|prescri\w*|you should take)\b")),
    ("secrecy", re.compile(r"\b(?:i won't tell anyone|our secret)\b")),
    ("invented news", re.compile(r"\b(?:breaking news|i heard on the news)\b")),
    ("claims to be human", re.compile(r"\b(?:i am human|i'm a real person)\b")),
)


@dataclass(frozen=True)
class ReplyCheck:
    ok: bool
    problems: tuple[str, ...]
    flags: tuple[str, ...]


def split_sentences(text: str) -> list[str]:
    """Sentences, not split after common abbreviations ("Mr.", "e.g.") or inside numbers ("3.5")."""
    sentences: list[str] = []
    start = 0
    for m in _SENTENCE_END.finditer(text):
        last_word = text[start : m.end()].split()[-1].lower()
        if last_word in _ABBREVIATIONS:
            continue
        sentences.append(text[start : m.end()].strip())
        start = m.end()
    rest = text[start:].strip()
    if rest:
        sentences.append(rest)
    return [s for s in sentences if s]


def check_reply(reply: str, note: RoomNote | None) -> ReplyCheck:
    body = _TRAILING_NOTE.sub("", reply)
    problems: list[str] = []
    sentences = split_sentences(body)
    if not sentences:
        problems.append("no spoken reply")
    if len(sentences) > MAX_SENTENCES:
        problems.append(f"{len(sentences)} sentences, more than {MAX_SENTENCES}")
    questions = sum(1 for s in sentences if s.endswith("?"))
    if questions > MAX_QUESTIONS:
        problems.append(f"{questions} questions, more than {MAX_QUESTIONS}")
    if "<note>" in body or "</note>" in body:
        problems.append("a memory note that is not at the very end")
    if OPEN in body or CLOSE in body or _KEY_LINE.search(body):
        problems.append("room-note tag or key line in the reply")
    if note is not None:
        problems.extend(_note_copies(body, note))
    return ReplyCheck(ok=not problems, problems=tuple(problems), flags=curator_flags(reply))


def curator_flags(reply: str) -> tuple[str, ...]:
    text = _plain_quotes(reply.lower())
    return tuple(name for name, pattern in _FLAGS if pattern.search(text))


def _note_copies(body: str, note: RoomNote) -> list[str]:
    entries = [object_text(o) for o in note.objects]
    problems = [f"exact note entry {e!r}" for e in entries if e in body]
    reply_grams = _grams(_words(body))
    for memory in note.notes:
        if _grams(_words(memory)) & reply_grams:
            problems.append(f"{COPY_WORDS}+ words copied from the memory note {memory!r}")
    return problems


def _words(text: str) -> list[str]:
    return _WORD.findall(_plain_quotes(text.lower()))


def _grams(words: list[str]) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + COPY_WORDS]) for i in range(len(words) - COPY_WORDS + 1)}


def _plain_quotes(text: str) -> str:
    return text.replace("’", "'").replace("‘", "'")
