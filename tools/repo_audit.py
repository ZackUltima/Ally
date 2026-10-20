"""Public-repository audit (D-52).

Fails when a tracked file is on the D-52 exclusion list (papers, data, backups,
editor files, course PDFs) or contains an e-mail address or a secret-looking
assignment. CI runs it on every push; the PO runs it before the first push.

Usage: python tools/repo_audit.py   (exit 1 if any problem is found)
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from collections.abc import Callable
from fnmatch import fnmatch
from pathlib import Path

logger = logging.getLogger("repo_audit")

EXCLUDED_PATTERNS: tuple[str, ...] = (
    "research/papers/*",
    "*.bak-*",
    ".mcp.json",
    ".roo/*",
    ".claude/settings.local.json",
    "temporary-for-srd-delete-if-done/*",
    "*/data/*",
    "*.mp4",
    "*.avi",
    "*.mov",
    "*.mkv",
    "*.npz",
    "*.npy",
    "*.parquet",
    "*.task",
    ".env",
    "*.env",
)

EMAIL = re.compile(
    r"[\w.+-]+(?:@|&#64;|&#x40;)(?:[\w-]+\.)+[a-z]{2,}\b", re.IGNORECASE
)
# Dataset-access contacts published on the datasets' own pages (D-41), kept on purpose.
ALLOWED_EMAILS: frozenset[str] = frozenset(
    {"facesdatabase@mpib-berlin.mpg.de", "queenie3@live.com"}
)
SECRET = re.compile(
    r"^\s*[A-Z0-9_]*(?:KEY|SECRET|TOKEN|PASSWORD)\s*[=:]\s*"
    r"(?:['\"][^'\"]+['\"]|[A-Za-z0-9_\-./+=]{3,})\s*(?:#.*)?$",
    re.MULTILINE,
)
MAX_TEXT_BYTES = 5_000_000


def _excluded(path: str) -> bool:
    return any(fnmatch(path, pattern) for pattern in EXCLUDED_PATTERNS)


def _has_private_email(text: str) -> bool:
    for match in EMAIL.finditer(text):
        address = re.sub(r"&#(?:64|x40);", "@", match.group(0), flags=re.IGNORECASE)
        if address.lower() not in ALLOWED_EMAILS:
            return True
    return False


def find_problems(paths: list[str], read_text: Callable[[str], str]) -> list[str]:
    """Return one message per tracked path that must not be published."""
    problems: list[str] = []
    for path in paths:
        if _excluded(path):
            problems.append(f"{path}: excluded from the public repository (D-52)")
            continue
        text = read_text(path)
        if _has_private_email(text):
            problems.append(f"{path}: contains an e-mail address")
        elif SECRET.search(text):
            problems.append(f"{path}: contains a secret-looking assignment")
    return problems


def _read_text(path: str) -> str:
    file = Path(path)
    try:
        if file.stat().st_size > MAX_TEXT_BYTES:
            return ""
        return file.read_text(encoding="utf-8", errors="ignore")
    except OSError as error:
        logger.error("cannot read %s: %s", path, error)
        raise


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        listed = subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        logger.error("git ls-files failed: %s", error)
        return 2
    paths = [line for line in listed.stdout.splitlines() if line]
    problems = find_problems(paths, _read_text)
    for problem in problems:
        logger.error(problem)
    logger.info("%d tracked files checked, %d problems", len(paths), len(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
