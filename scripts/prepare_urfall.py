"""Download and lay out UR Fall (CC BY-NC-SA 4.0) under data/urfall/ with subject IDs for
leave-one-subject-out.

Sprint: S2. Not implemented yet; exits 2 so Makefile targets fail loudly rather than silently.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_known_args()
    print("prepare_urfall: not implemented until S2", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
