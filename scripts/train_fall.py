"""Train rule / LightGBM / GRU fall classifiers with LOSO-CV on UR Fall keypoints; fixed seed;
results/fall/<date>/.

Sprint: S2. Not implemented yet; exits 2 so Makefile targets fail loudly rather than silently.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_known_args()
    print("train_fall: not implemented until S2", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
