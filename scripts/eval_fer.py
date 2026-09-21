"""FER-2013 test accuracy, macro-F1, confusion matrix.

Sprint: S5. Not implemented yet; exits 2 so Makefile targets fail loudly rather than silently.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_known_args()
    print("eval_fer: not implemented until S5", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
