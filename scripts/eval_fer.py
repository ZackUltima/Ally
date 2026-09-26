"""FER accuracy, macro-F1, confusion matrix on the public test split, and by age group on an older-adult set
(FACES or ElderReact) so the age gap is reported (ADR-013, result table 3a).

Sprint: S2. Not implemented yet; exits 2 so Makefile targets fail loudly rather than silently.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_known_args()
    print("eval_fer: not implemented until S2", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
