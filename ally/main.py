"""Entry point: starts workers over bounded drop-oldest queues (capture → perception → fusion → agent).

`--replay <session>` swaps the webcam for a recording made by scripts/record_session.py (ADR-004); the same
code path runs from perception onward. Worker wiring lands in Sprint 1.
"""

from __future__ import annotations

import argparse
import sys

from ally import __version__
from ally.config import get_settings


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ally", description=__doc__.split("\n")[0])
    p.add_argument("--replay", metavar="SESSION", help="replace the webcam with data/sessions/<SESSION>/")
    p.add_argument("--version", action="version", version=f"ally {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    mode = f"replay:{args.replay}" if args.replay else "live"
    print(
        f"ally {__version__} — mode={mode}"
        f" capture={settings.ally_capture_width}x{settings.ally_capture_height}"
        f" target_fps={settings.ally_target_fps}"
    )
    print("pipeline workers are not wired yet (Sprint 1). Nothing was started.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
