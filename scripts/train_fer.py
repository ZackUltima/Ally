"""Fine-tune a small pretrained FER model (FER-2013 and/or AffectNet; fallback: pretrained weights);
fixed seed; results/fer/<date>/.

ADR-013. Dataset licences recorded in decisions.md first (P-3).
Sprint: S2. Not implemented yet; exits 2 so Makefile targets fail loudly rather than silently.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_known_args()
    print("train_fer: not implemented until S2", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
