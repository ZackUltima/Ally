"""Record keypoints + mic audio + event log to data/sessions/<name>/ (never raw frames) for --replay
(ADR-004).

Sprint: S1. Not implemented yet; exits 2 so Makefile targets fail loudly rather than silently.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_known_args()
    print("record_session: not implemented until S1", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
