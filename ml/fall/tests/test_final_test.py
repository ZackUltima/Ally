import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

import ally_fall
from ally_fall.baseline import Thresholds
from ally_fall.final_test import (
    Breakdown,
    FinalTest,
    breakdown,
    dim_video_mask,
    evaluate,
    parse_dim_videos,
    precision_at,
    result_json,
    thresholds_from,
    write_breakdown,
    write_reports,
)
from ally_fall.models import build_model
from ally_fall.train import RunConfig

FIXTURES = Path(__file__).parent / "fixtures" / "omnifall"
WALK, FALL, FALLEN = 0, 1, 2
DIM_SENTENCE = (
    "Proposed rule: a test video is dim if its mean brightness is below 78 (the midpoint of "
    "the gap); that gives exactly person 8's 10 videos (FallBackwards, FallForward, FallLeft, "
    "FallRight, FallSitting, Hop, Kneel, Pickupobject, SitDown and Walk S8). Next sentence."
)
# (dataset, subject, video id): nine people, as in the real test split; person 8 is dim.
VIDEOS = [
    ("up_fall", 2, "Subject2/Activity1/Trial1/cam1"),
    ("up_fall", 4, "Subject4/Activity1/Trial1/cam1"),
    ("up_fall", 15, "Subject15/Activity1/Trial1/cam1"),
    ("up_fall", 16, "Subject16/Activity1/Trial1/cam1"),
    ("caucafall", 8, "forward/FallForwardS8"),
    ("caucafall", 9, "forward/FallForwardS9"),
    ("le2i", 2, "Coffee_room_01/video_1"),
    ("le2i", 7, "Home_01/video_2"),
    ("GMDCSA24", 4, "Subject_4/Fall/01"),
]
TICKS = 80


def _video(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Walking, a fast drop (fall), then lying still (fallen)."""
    features = np.zeros((TICKS, 33, 3), dtype=np.float32)
    base_y = np.full(TICKS, 0.3)
    base_y[40:46] = np.linspace(0.3, 0.7, 6)
    base_y[46:] = 0.7
    features[:, :, 0] = rng.uniform(0.4, 0.6, size=(TICKS, 33))
    features[:, :, 1] = base_y[:, None] + rng.uniform(-0.05, 0.05, size=(TICKS, 33))
    features[:, :, 2] = 1.0
    labels = np.full(TICKS, WALK, dtype=np.int64)
    labels[40:46] = FALL
    labels[46:] = FALLEN
    return features, labels


def _write_test_split(path: Path, blank_video: int | None = None) -> np.ndarray:
    """Writes a built split of the nine people; `blank_video` loses its pose in ticks 0-39."""
    rng = np.random.default_rng(3)
    parts = [_video(rng) for _ in VIDEOS]
    if blank_video is not None:
        parts[blank_video][0][:40] = 0.0
    starts = np.arange(len(VIDEOS)) * TICKS
    window_end = np.concatenate([s + np.arange(29, TICKS) for s in starts])
    labels = np.concatenate([lab for _, lab in parts])
    np.savez(path, features=np.concatenate([f for f, _ in parts]), labels=labels,
             video_start=starts, video_len=np.full(len(VIDEOS), TICKS),
             subject=np.array([s for _, s, _ in VIDEOS]),
             dataset=np.array([d for d, _, _ in VIDEOS]),
             video_id=np.array([v for _, _, v in VIDEOS]), window_end=window_end,
             y=labels[window_end])
    return window_end // TICKS


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    (tmp_path / "omnifall" / "labels").mkdir(parents=True)
    shutil.copy(FIXTURES / "labels" / "label2id.csv", tmp_path / "omnifall" / "labels")
    (tmp_path / "built" / "lite").mkdir(parents=True)
    return tmp_path


def _cfg() -> RunConfig:
    return RunConfig(run_id="t", model="tcn", label_set="omnifall16", beta=0.99, hidden=8)


def _weights(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    path = tmp_path / "best.pt"
    torch.save(build_model("tcn", 16, 8).state_dict(), path)
    return path


def _oracle(data_root: Path, wrong_video: int | None = None):  # noqa: ANN202
    """An INT8 stand-in that answers with the true label, except on one video where it says walk."""
    with np.load(data_root / "built" / "lite" / "test.npz") as data:
        y = data["y"].copy()
        video = data["window_end"] // TICKS
    if wrong_video is not None:
        y[video == wrong_video] = WALK

    def predict(windows: np.ndarray) -> np.ndarray:
        assert len(windows) == len(y)
        return y

    return predict


def _run(data_root: Path, tmp_path: Path, wrong_video: int | None = 4) -> FinalTest:
    _write_test_split(data_root / "built" / "lite" / "test.npz")
    dim = parse_dim_videos(DIM_SENTENCE)
    return evaluate(_cfg(), data_root, _weights(tmp_path), _oracle(data_root, wrong_video),
                    Thresholds(2.25, 0.3), dim, split="test")


def test_refuses_when_the_lock_exists_and_never_opens_the_split(data_root: Path,
                                                               tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "test.json").write_text("{}")
    (data_root / "built" / "lite" / "test.npz").write_bytes(b"must not be opened")
    result = FinalTest.empty()
    with pytest.raises(FileExistsError, match="test.json"):
        write_reports(result, out, "test")
    with pytest.raises(FileExistsError, match="test.json"):
        evaluate(_cfg(), data_root, _weights(tmp_path), lambda w: w, Thresholds(1.0, 0.1),
                 frozenset(), split="test", lock=out / "test.json")


def test_writes_the_lock_and_report(data_root: Path, tmp_path: Path) -> None:
    result = _run(data_root, tmp_path)
    out = tmp_path / "out"
    write_reports(result, out, "test")
    lock = json.loads((out / "test.json").read_text())
    text = (out / "test.md").read_text()
    assert lock["split"] == "test" and lock["people"]["all"] == 9
    assert "INT8" in text and "without" in text.lower() and "threshold rule" in text.lower()
    with pytest.raises(FileExistsError):
        write_reports(result, out, "test")


def test_rehearsal_on_another_split_is_not_locked(data_root: Path, tmp_path: Path) -> None:
    result = _run(data_root, tmp_path)
    out = tmp_path / "out"
    write_reports(result, out, "val")
    write_reports(result, out, "val")
    assert (out / "rehearsal-val.json").exists() and not (out / "test.json").exists()
    text = (out / "rehearsal-val.md").read_text()
    assert "All val people" in text and "All test people" not in text


def test_dim_list_is_read_from_the_f1_report() -> None:
    names = parse_dim_videos("intro\n" + DIM_SENTENCE)
    assert len(names) == 10
    assert {"FallBackwardsS8", "WalkS8", "PickupobjectS8", "SitDownS8"} <= names
    with pytest.raises(ValueError, match="dim"):
        parse_dim_videos("no such sentence here")
    with pytest.raises(ValueError, match="10"):
        parse_dim_videos(DIM_SENTENCE.replace("10 videos", "11 videos"))


def test_dim_mask_needs_dataset_and_name() -> None:
    names = parse_dim_videos(DIM_SENTENCE)
    datasets = ["caucafall", "caucafall", "le2i", "caucafall"]
    ids = ["forward/FallForwardS8", "forward/FallForwardS9", "adl/HopS8", "adl/HopS8"]
    assert dim_video_mask(ids, datasets, names).tolist() == [True, False, False, True]


def test_scores_with_and_without_the_dim_videos(data_root: Path, tmp_path: Path) -> None:
    result = _run(data_root, tmp_path, wrong_video=4)  # video 4 is caucafall person 8
    with_dim, without = result.conditions["all"], result.conditions["no_dim"]
    assert (with_dim.people, without.people) == (9, 8)
    assert without.windows == with_dim.windows - (TICKS - 29)
    # the INT8 stand-in is wrong only on the dim video, so dropping it makes it perfect
    assert with_dim.int8["fall_class"].val.recall < 1.0
    assert without.int8["fall_class"].val.recall == 1.0
    assert without.int8["fall_class"].val.precision == 1.0


def test_reports_fall_label_alone_and_fall_class(data_root: Path, tmp_path: Path) -> None:
    cond = _run(data_root, tmp_path, wrong_video=None).conditions["all"]
    both, alone = cond.int8["fall_class"].val, cond.int8["fall_only"].val
    assert both.tp + both.fn == 9 * (TICKS - 40)  # fall and fallen windows, per video
    assert alone.tp + alone.fn == 9 * 6 and alone.tp + alone.fn < both.tp + both.fn
    assert alone.fp == 0 and alone.fn == 0  # the stand-in answers every label correctly


def test_intervals_are_by_person_over_nine_people(data_root: Path, tmp_path: Path) -> None:
    cond = _run(data_root, tmp_path).conditions["all"]
    assert cond.people == 9 and cond.videos == 9
    for key in ("recall", "precision", "f1"):
        low, high = cond.int8["fall_class"].intervals[key]
        assert low <= high
    # one dim person out of nine: resampling people can reach the wrong video, so the
    # interval is wide, which resampling windows (thousands) never would be
    low, high = cond.int8["fall_class"].intervals["recall"]
    assert low < cond.int8["fall_class"].val.recall <= high


def test_threshold_rule_scored_on_the_same_people(data_root: Path, tmp_path: Path) -> None:
    result = _run(data_root, tmp_path)
    for name, cond in result.conditions.items():
        assert cond.rule.val.tp + cond.rule.val.fp + cond.rule.val.fn + cond.rule.val.tn == \
            cond.windows, name
        assert cond.int8["fall_class"].val.tp + cond.int8["fall_class"].val.fn == \
            cond.rule.val.tp + cond.rule.val.fn


def test_float_and_int8_agreement_is_reported(data_root: Path, tmp_path: Path) -> None:
    result = _run(data_root, tmp_path)
    assert 0.0 <= result.fall_agreement <= 1.0 and 0.0 <= result.label_agreement <= 1.0


def test_thresholds_come_from_the_f2_baseline_file(tmp_path: Path) -> None:
    path = tmp_path / "baseline-lite.json"
    path.write_text(json.dumps({"thresholds": {"speed": 2.25, "angle": 0.3}}))
    assert thresholds_from(path) == Thresholds(2.25, 0.3)


def test_no_other_module_imports_the_test_loader() -> None:
    src = Path(ally_fall.__file__).parent
    forbidden = ("import final_test", "ally_fall.final_test", "from ally_fall import final_test",
                 "test.npz")
    offenders = [p.name for p in src.rglob("*.py") if p.name != "final_test.py"
                 and any(word in p.read_text(encoding="utf-8") for word in forbidden)]
    assert offenders == []


# ---------------------------------------------------------------------- the error analysis

def _flagging(data_root: Path, false_alarm_video: int | None):  # noqa: ANN202
    """An INT8 stand-in that is right everywhere except that it calls one video's walking a fall."""
    with np.load(data_root / "built" / "lite" / "test.npz") as data:
        y = data["y"].copy()
        video = data["window_end"] // TICKS
    if false_alarm_video is not None:
        y[(video == false_alarm_video) & (y == WALK)] = FALL
    return lambda windows: y


def _analysis(data_root: Path, tmp_path: Path, flag: int | None = None,
              blank: int | None = None) -> Breakdown:
    built = data_root / "built" / "lite"
    _write_test_split(built / "test.npz", blank_video=blank)
    _write_test_split(built / "val.npz")
    predict = _flagging(data_root, flag)
    weights = _weights(tmp_path)
    reference = result_json(evaluate(_cfg(), data_root, weights, predict, Thresholds(2.25, 0.3),
                                     frozenset(), split="test"), "test")
    return breakdown(_cfg(), data_root, predict, reference)


def test_breakdown_stops_when_it_does_not_reproduce_the_test_result(data_root: Path,
                                                                    tmp_path: Path) -> None:
    built = data_root / "built" / "lite"
    _write_test_split(built / "test.npz")
    _write_test_split(built / "val.npz")
    weights = _weights(tmp_path)
    reference = result_json(evaluate(_cfg(), data_root, weights, _flagging(data_root, None),
                                     Thresholds(2.25, 0.3), frozenset()), "test")
    with pytest.raises(ValueError, match="does not reproduce"):
        breakdown(_cfg(), data_root, _flagging(data_root, 0), reference)


def test_breakdown_groups_errors_by_dataset_and_person(data_root: Path, tmp_path: Path) -> None:
    b = _analysis(data_root, tmp_path, flag=0)  # video 0 is up_fall person 2
    assert set(b.by_dataset) == {"up_fall", "caucafall", "le2i", "GMDCSA24"}
    assert len(b.by_person) == 9
    assert b.by_person["up_fall/2"].fp == 11 and b.by_dataset["up_fall"].fp == 11
    assert b.by_dataset["caucafall"].fp == 0 and b.by_person["up_fall/2"].false_alerts == 1
    assert sum(s.windows for s in b.by_dataset.values()) == 9 * (TICKS - 29)


def test_false_positives_are_listed_by_true_label(data_root: Path, tmp_path: Path) -> None:
    rows = {r.label: r for r in _analysis(data_root, tmp_path, flag=0).false_positive_labels}
    assert rows["walk"].flagged == 11 and rows["walk"].windows == 9 * 11
    assert all(r.flagged == 0 for label, r in rows.items() if label != "walk")


def test_pose_loss_bins_show_where_the_false_alarms_are(data_root: Path, tmp_path: Path) -> None:
    b = _analysis(data_root, tmp_path, flag=1, blank=1)
    expected = sum(1 for end in range(29, TICKS)
                   if (min(end, 39) - (end - 29) + 1 if end - 29 <= 39 else 0) / 30 > 0.5)
    over = {bin_.name: bin_ for bin_ in b.pose_bins}["over half of the ticks"]
    assert over.windows == expected == 25
    assert over.false_positives == 11 and over.share_of_false_positives == 1.0
    assert sum(bin_.windows for bin_ in b.pose_bins) == 9 * (TICKS - 29)


def test_precision_at_another_prevalence() -> None:
    assert precision_at(recall=0.8, specificity=0.9, prevalence=0.5) == pytest.approx(8 / 9)
    assert precision_at(recall=1.0, specificity=1.0, prevalence=0.0) == 0.0


def test_breakdown_is_labelled_as_analysis_not_a_score(data_root: Path, tmp_path: Path) -> None:
    b = _analysis(data_root, tmp_path, flag=0)
    out = tmp_path / "out"
    write_breakdown(b, out)
    text = (out / "test-breakdown.md").read_text()
    assert "error analysis" in text.lower() and "not a new score" in text.lower()
    assert json.loads((out / "test-breakdown.json").read_text())["kind"] == "error_analysis"
    assert not (out / "test.json").exists()
