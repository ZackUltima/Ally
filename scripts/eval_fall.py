"""Evaluate saved fall models: recall (mean +/- 95% CI), specificity, F1, suspected FA/h, latency;
ONNX export check.

Sprint: S2. Not implemented yet; exits 2 so Makefile targets fail loudly rather than silently.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_known_args()
    print("eval_fall: not implemented until S2", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
