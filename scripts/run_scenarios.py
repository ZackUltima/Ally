"""Run tests/scenarios/*.yaml through the agent with a fake speech layer; report FSM decision accuracy
and latency; --repeat N.

Sprint: S4. Not implemented yet; exits 2 so Makefile targets fail loudly rather than silently.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_known_args()
    print("run_scenarios: not implemented until S4", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
