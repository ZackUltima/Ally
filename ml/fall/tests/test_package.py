"""Package smoke test."""

import ally_fall


def test_version() -> None:
    assert ally_fall.__version__ == "0.1.0"
