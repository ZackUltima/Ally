"""Tests for loading OmniFall labels (D-54). The fixture rows are made up."""

from pathlib import Path
from unittest.mock import patch

from ally_fall import omnifall
from ally_fall.ticks import Segment

FIXTURES = Path(__file__).parent / "fixtures" / "omnifall"


def test_label_names() -> None:
    names = omnifall.label_names(FIXTURES)
    assert len(names) == 16
    assert names[1] == "fall" and names[2] == "fallen"
    assert names[5] == "lie_down" and names[6] == "lying"


def test_load_table_columns() -> None:
    table = omnifall.load_table(FIXTURES / "labels" / "sample.csv")
    assert list(table.columns) == ["path", "label", "start", "end", "subject", "cam", "dataset"]
    assert table["label"].dtype.kind == "i" and table["start"].dtype.kind == "f"


def test_segments_by_video() -> None:
    table = omnifall.load_table(FIXTURES / "labels" / "sample.csv")
    segments, conflicts = omnifall.segments_by_video(table, "caucafall")
    assert set(segments) == {("Fall/S1_FallBackwards", 1), ("ADL/S2_Walk", 1)}
    assert conflicts == {}
    assert segments[("Fall/S1_FallBackwards", 1)] == [
        Segment(0.0, 2.0, 0),
        Segment(2.0, 3.1, 1),
        Segment(3.1, 6.0, 2),
    ]


def test_small_overlap_trimmed() -> None:
    # PO 2026-10-20, option A: an overlap up to one tick (0.1 s) is rounding; the later
    # label takes over from its start.
    table = omnifall.load_table(FIXTURES / "labels" / "overlap_small.csv")
    segments, conflicts = omnifall.segments_by_video(table, "caucafall")
    assert conflicts == {}
    assert segments[("Fall/S1_Rounding", 1)] == [Segment(0.0, 2.0, 0), Segment(2.0, 3.0, 1)]
    assert segments[("Fall/S1_AtLimit", 1)] == [Segment(0.0, 2.0, 0), Segment(2.0, 3.0, 1)]


def test_large_overlap_reported() -> None:
    # Option A: more than 0.1 s is a labelling conflict; the video is left out and reported.
    table = omnifall.load_table(FIXTURES / "labels" / "overlap.csv")
    segments, conflicts = omnifall.segments_by_video(table, "caucafall")
    assert ("Fall/S1_Overlap", 1) not in segments
    assert conflicts == {("Fall/S1_Overlap", 1): "overlapping labels (0.50 s)"}


def test_split_paths() -> None:
    splits = omnifall.split_paths(FIXTURES, "caucafall")
    assert splits["train"] == {"Fall/S1_FallBackwards"}
    assert splits["test"] == {"ADL/S2_Walk"}
    assert splits["val"] == set()


def test_download_labels_only_fetches_label_files(tmp_path: Path) -> None:
    with patch.object(omnifall, "snapshot_download") as fake:
        fake.return_value = str(tmp_path / "omnifall")
        result = omnifall.download_labels(tmp_path)
    kwargs = fake.call_args.kwargs
    assert kwargs["repo_id"] == "simplexsigil2/omnifall"
    assert kwargs["repo_type"] == "dataset"
    # Pinned to the commit checked on 2026-10-20 (PO, D-54), so an upstream change cannot
    # alter our labels or splits unnoticed.
    assert kwargs["revision"] == "83572a37b9e3081df8c06a56874b1d1f2a19386c"
    assert all(not p.startswith("data_files") for p in kwargs["allow_patterns"])
    assert "labels/*.csv" in kwargs["allow_patterns"]
    assert result == tmp_path / "omnifall"
