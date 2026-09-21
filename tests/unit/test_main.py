from __future__ import annotations

import pytest

from ally.main import build_parser, main


def test_replay_flag_parses():
    assert build_parser().parse_args(["--replay", "fall-01"]).replay == "fall-01"


def test_main_reports_unwired_pipeline(clean_env, capsys):
    assert main(["--replay", "demo"]) == 2
    out, err = capsys.readouterr()
    assert "mode=replay:demo" in out
    assert "640x480" in out
    assert "Sprint 1" in err


def test_version_flag():
    with pytest.raises(SystemExit) as e:
        build_parser().parse_args(["--version"])
    assert e.value.code == 0
