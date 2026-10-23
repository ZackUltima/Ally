"""Tests for the build command (technical plan §2, §10). All data here is made up."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ally_fall.build import EXCLUDED, EXCLUDED_CAMERAS, build, keypoint_path, window_features
from ally_fall.keypoints import KeypointSeq, save_seq
from ally_fall.ticks import Segment
from ally_fall.windows import make_windows

NAMES = [
    "walk", "fall", "fallen", "sit_down", "sitting", "lie_down", "lying", "stand_up",
    "standing", "other", "kneel_down", "kneeling", "squat_down", "squatting", "crawl", "jump",
]
# GMDCSA24 cross-subject split: train 1, 3; val 2; test 4 (D-54).
VIDEOS = {
    "Subject_1/Fall/01": 1,
    "Subject_3/Fall/01": 3,
    "Subject_2/Fall/01": 2,
    "Subject_4/Fall/01": 4,
}
SEGMENTS = [Segment(0.0, 2.0, 0), Segment(2.0, 2.5, 1), Segment(2.5, 4.0, 2)]


def _seq(video_id: str, subject: int, frames: int = 100, fps: float = 25.0) -> KeypointSeq:
    rng = np.random.default_rng(subject)
    return KeypointSeq(
        dataset="GMDCSA24",
        video_id=video_id,
        subject=subject,
        cam=1,
        pose_model="lite",
        frame_times_s=np.arange(frames) / fps,
        xy=rng.uniform(0.1, 0.9, (frames, 33, 2)).astype(np.float32),
        vis=np.full((frames, 33), 0.9, dtype=np.float32),
    )


@pytest.fixture()
def data_root(tmp_path: Path) -> Path:
    labels = tmp_path / "omnifall" / "labels"
    labels.mkdir(parents=True)
    (labels / "label2id.csv").write_text(
        "label,id\n" + "".join(f"{n},{i}\n" for i, n in enumerate(NAMES))
    )
    rows = ["path,label,start,end,subject,cam,dataset"]
    for video, subject in [*VIDEOS.items(), ("Subject_3/Fall/09", 3)]:
        rows += [f"{video},{s.label},{s.start_s},{s.end_s},{subject},1,GMDCSA24" for s in SEGMENTS]
    (labels / "GMDCSA24.csv").write_text("\n".join(rows) + "\n")
    for video, subject in VIDEOS.items():
        save_seq(_seq(video, subject), keypoint_path(tmp_path, "lite", "GMDCSA24", video, 1))
    unlabelled = "Subject_1/ADL/99"
    save_seq(_seq(unlabelled, 1), keypoint_path(tmp_path, "lite", "GMDCSA24", unlabelled, 1))
    return tmp_path


def _load(root: Path, split: str) -> dict[str, np.ndarray]:
    with np.load(root / "built" / "lite" / f"{split}.npz", allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def test_no_test_person_in_train(data_root: Path) -> None:
    build(data_root, ["GMDCSA24"])
    train, val, test = (_load(data_root, s) for s in ("train", "val", "test"))
    assert set(train["subject"].tolist()) == {1, 3}
    assert not set(train["subject"].tolist()) & (
        set(val["subject"].tolist()) | set(test["subject"].tolist())
    )
    assert set(test["subject"].tolist()) == {4}


def test_label_counts_saved(data_root: Path) -> None:
    build(data_root, ["GMDCSA24"])
    counts = json.loads((data_root / "built" / "lite" / "label_counts_train.json").read_text())
    y = _load(data_root, "train")["y"]
    assert list(counts) == NAMES
    assert counts == {name: int((y == i).sum()) for i, name in enumerate(NAMES)}


def test_windows_match_make_windows(data_root: Path) -> None:
    build(data_root, ["GMDCSA24"])
    train = _load(data_root, "train")
    x = window_features(train["features"], train["window_end"])
    expected_x, expected_y = [], []
    for start, length in zip(train["video_start"], train["video_len"], strict=True):
        vx, vy = make_windows(
            train["features"][start : start + length], train["labels"][start : start + length]
        )
        expected_x.append(vx)
        expected_y.append(vy)
    assert np.array_equal(x, np.concatenate(expected_x))
    assert np.array_equal(train["y"], np.concatenate(expected_y))
    assert x.shape[1:] == (30, 33, 3)


def test_ticks_are_10_fps(data_root: Path) -> None:
    report = build(data_root, ["GMDCSA24"])
    # 100 frames at 25 fps = 0 .. 3.96 s -> ticks 0.0 .. 3.9 s = 40 ticks, 11 windows per video;
    # two training videos, and no window may run across the boundary between them.
    assert report.splits["train"].videos == 2
    assert report.splits["train"].ticks == 80
    assert report.splits["train"].windows == 22


def test_video_without_labels_skipped(data_root: Path) -> None:
    report = build(data_root, ["GMDCSA24"])
    assert ("GMDCSA24", "Subject_1/ADL/99", "no labels") in report.skipped


def test_label_row_without_video_reported(data_root: Path) -> None:
    report = build(data_root, ["GMDCSA24"])
    assert ("GMDCSA24", "Subject_3/Fall/09", "no keypoints") in report.skipped


def test_video_with_label_conflict_skipped(data_root: Path) -> None:
    conflicted = "Subject_1/Fall/05"
    with (data_root / "omnifall" / "labels" / "GMDCSA24.csv").open("a") as file:
        file.write(f"{conflicted},7,0.0,3.0,1,1,GMDCSA24\n{conflicted},4,2.0,4.0,1,1,GMDCSA24\n")
    save_seq(_seq(conflicted, 1), keypoint_path(data_root, "lite", "GMDCSA24", conflicted, 1))
    report = build(data_root, ["GMDCSA24"])
    reasons = [reason for _, video, reason in report.skipped if video == conflicted]
    assert reasons == ["overlapping labels (1.00 s)"]
    assert report.splits["train"].videos == 2


def test_excluded_video_skipped_with_reason(data_root: Path) -> None:
    reason = "person out of the picture"
    excluded = {("GMDCSA24", "Subject_3/Fall/01", 1): reason}
    report = build(data_root, ["GMDCSA24"], excluded=excluded)
    assert ("GMDCSA24", "Subject_3/Fall/01", reason) in report.skipped
    assert not [s for s in report.skipped if s[1] == "Subject_3/Fall/01" and s[2] == "no keypoints"]
    assert report.splits["train"].videos == 1
    assert set(_load(data_root, "train")["subject"].tolist()) == {1}


def test_excluded_camera_skipped_with_reason(data_root: Path) -> None:
    video = "Subject_1/Fall/01"
    with (data_root / "omnifall" / "labels" / "GMDCSA24.csv").open("a") as file:
        file.write("".join(f"{video},{s.label},{s.start_s},{s.end_s},1,2,GMDCSA24\n"
                           for s in SEGMENTS))
    save_seq(replace(_seq(video, 1), cam=2), keypoint_path(data_root, "lite", "GMDCSA24", video, 2))
    assert build(data_root, ["GMDCSA24"], excluded_cameras={}).splits["train"].videos == 3
    report = build(data_root, ["GMDCSA24"], excluded_cameras={("GMDCSA24", 2): "camera out"})
    assert ("GMDCSA24", video, "camera out") in report.skipped
    assert report.splits["train"].videos == 2


def test_excluded_cameras_is_up_fall_camera_2() -> None:
    # D-54 (PO 2026-10-22, option A): in UP-Fall camera 2 the pose is often on a seated bystander.
    assert set(EXCLUDED_CAMERAS) == {("up_fall", 2)}
    assert "D-54" in EXCLUDED_CAMERAS[("up_fall", 2)]


def test_excluded_list_is_up_fall_lying_camera_1() -> None:
    # D-54 (PO 2026-10-22): camera 1 of Activity 11 (lying) for subjects 5, 11, 12 and 15, where
    # the person lies at the bottom edge of the picture and a pose is found in only 0-5% of frames.
    expected = {
        ("up_fall", f"Subject{s}/Activity11/Trial{t}/Subject{s}Activity11Trial{t}Camera1", 1)
        for s in (5, 11, 12, 15)
        for t in (1, 2, 3)
    }
    assert set(EXCLUDED) == expected
    assert all("D-54" in reason for reason in EXCLUDED.values())


def test_points_outside_image_marked_missing(data_root: Path) -> None:
    seq = _seq("Subject_4/Fall/01", 4)
    seq.xy[:, 0, 0] = 1.5
    save_seq(seq, keypoint_path(data_root, "lite", "GMDCSA24", "Subject_4/Fall/01", 1))
    build(data_root, ["GMDCSA24"])
    test = _load(data_root, "test")
    assert not test["features"][:, 0, 2].any()
    assert test["features"][:, 1, 2].all()


def test_cli_builds_and_returns_0(data_root: Path) -> None:
    from ally_fall.build import main

    assert main(["--data-root", str(data_root), "--datasets", "GMDCSA24"]) == 0
    assert (data_root / "built" / "lite" / "train.npz").exists()


def test_cli_returns_1_when_labels_missing(tmp_path: Path) -> None:
    from ally_fall.build import main

    assert main(["--data-root", str(tmp_path), "--datasets", "GMDCSA24"]) == 1
