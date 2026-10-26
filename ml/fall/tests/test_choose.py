"""Tests for the F3b model-choice rule (F3b Task 1, D-68 decision 1)."""

import json
from pathlib import Path

import pytest

import ally_fall.choose as choose
from ally_fall.choose import Candidate

PT = "stgcn_pt"
TCN = "tcn"
FALLS = 57


def _cand(name: str, f1: float, detected: float, *, spread: float = 0.0) -> Candidate:
    """Three seeds around the given means (so the mean is exact)."""
    f1s = {0: f1 - spread, 1: f1, 2: f1 + spread}
    found = {0: detected - 1, 1: detected, 2: detected + 1}
    return Candidate(
        name=name,
        f1=f1s,
        detected={s: int(v) for s, v in found.items()},
        missed={s: FALLS - int(v) for s, v in found.items()},
        false_alerts={0: 20, 1: 20, 2: 20},
        parameters=3_000_000 if name == PT else 50_000,
        recall={s: f1 for s in f1s},
        precision={s: f1 for s in f1s},
    )


def test_falls_found_decides_when_two_apart() -> None:
    # the TCN finds 3 more falls although its window F1 is lower
    pick = choose.choose(_cand(PT, 0.90, 51), _cand(TCN, 0.86, 54))
    assert (pick.winner, pick.branch) == (TCN, "falls_found")
    # and the other way round: the pretrained model finds 3 more and is clearly better
    pick = choose.choose(_cand(PT, 0.92, 56), _cand(TCN, 0.86, 53))
    assert (pick.winner, pick.branch) == (PT, "falls_found")


def test_exactly_two_falls_apart_counts_as_two_apart() -> None:
    pick = choose.choose(_cand(PT, 0.92, 52), _cand(TCN, 0.86, 54))
    assert (pick.winner, pick.branch) == (TCN, "falls_found")


def test_window_f1_decides_when_under_two_apart() -> None:
    pick = choose.choose(_cand(PT, 0.92, 52), _cand(TCN, 0.86, 53))
    assert (pick.winner, pick.branch) == (PT, "window_f1")
    pick = choose.choose(_cand(PT, 0.86, 53), _cand(TCN, 0.92, 52))
    assert (pick.winner, pick.branch) == (TCN, "window_f1")


def test_pretrained_needs_point_02_f1_over_tcn() -> None:
    pick = choose.choose(_cand(PT, 0.879, 53), _cand(TCN, 0.860, 53))
    assert (pick.winner, pick.branch) == (TCN, "not_clearly_better")
    pick = choose.choose(_cand(PT, 0.880, 53), _cand(TCN, 0.860, 53))
    assert (pick.winner, pick.branch) == (PT, "window_f1")


def test_tie_takes_the_smaller_model() -> None:
    pick = choose.choose(_cand(PT, 0.87, 53), _cand(TCN, 0.87, 53))
    assert pick.winner == TCN


def test_int8_gate_failure_takes_the_other_model() -> None:
    pt, tcn = _cand(PT, 0.92, 53), _cand(TCN, 0.86, 53)
    pick = choose.choose(pt, tcn, {PT: False, TCN: True})
    assert (pick.winner, pick.branch, pick.provisional) == (TCN, "int8_gate", False)
    pick = choose.choose(_cand(PT, 0.86, 56), _cand(TCN, 0.80, 50), {PT: True, TCN: False})
    assert (pick.winner, pick.branch) == (PT, "falls_found")
    pick = choose.choose(_cand(PT, 0.86, 50), _cand(TCN, 0.80, 56), {PT: True, TCN: False})
    assert (pick.winner, pick.branch) == (PT, "int8_gate")


def test_when_neither_passes_the_rule_winner_stays_and_the_po_is_told() -> None:
    pick = choose.choose(_cand(PT, 0.92, 53), _cand(TCN, 0.86, 53), {PT: False, TCN: False})
    assert (pick.winner, pick.branch) == (PT, "int8_gate_none_passed")
    assert "PO" in pick.why


def test_without_int8_results_the_choice_is_provisional() -> None:
    assert choose.choose(_cand(PT, 0.92, 53), _cand(TCN, 0.86, 53)).provisional
    assert choose.choose(_cand(PT, 0.92, 53), _cand(TCN, 0.86, 53), {}).provisional
    ok = choose.choose(_cand(PT, 0.92, 53), _cand(TCN, 0.86, 53), {PT: True})
    assert not ok.provisional


@pytest.mark.parametrize(
    ("pt_f1", "pt_found", "tcn_f1", "tcn_found", "int8", "winner", "branch"),
    [
        # step (i) comes before step (ii): a higher F1 never overrides 2+ more falls found
        (0.95, 50, 0.80, 52, None, TCN, "falls_found"),
        # step (ii) when under 2 apart
        (0.95, 52, 0.80, 53, None, PT, "window_f1"),
        # step (iv) comes after (i) and (ii)
        (0.89, 52, 0.88, 53, None, TCN, "not_clearly_better"),
        (0.89, 56, 0.88, 53, None, TCN, "not_clearly_better"),
        # step (iii) comes last and replaces the winner
        (0.95, 52, 0.80, 53, {PT: False, TCN: True}, TCN, "int8_gate"),
        (0.80, 52, 0.95, 53, {PT: True, TCN: False}, PT, "int8_gate"),
    ],
)
def test_rule_is_applied_in_the_stated_order(
    pt_f1: float, pt_found: float, tcn_f1: float, tcn_found: float,
    int8: dict[str, bool] | None, winner: str, branch: str,
) -> None:
    pick = choose.choose(_cand(PT, pt_f1, pt_found), _cand(TCN, tcn_f1, tcn_found), int8)
    assert (pick.winner, pick.branch) == (winner, branch)
    assert pick.why


def _run(results: Path, model: str, seed: int, f1: float, found: int, *, pose: str = "lite",
         beta: float = 0.999, label_set: str = "omnifall16", pretrained: bool | None = None,
         extra: dict[str, object] | None = None, name: str | None = None) -> Path:
    """One run's JSON, with the fields `train.results_json` writes that the rule reads."""
    pretrained = model == PT if pretrained is None else pretrained
    run_id = name or f"{model}-{label_set}-b{beta:g}-s{seed}-{pose}-{int(pretrained)}"
    results.mkdir(parents=True, exist_ok=True)
    data: dict[str, object] = {
        "run_id": run_id,
        "config": {"run_id": run_id, "model": model, "label_set": label_set, "beta": beta,
                   "pose_model": pose, "seed": seed, "hidden": 8,
                   "checkpoint": "models/x.pth" if pretrained else ""},
        "val": {"f1": f1, "recall": f1 + 0.01, "precision": f1 - 0.01},
        "events": {"detected": found, "missed": FALLS - found, "false_alerts": 20},
        "train_counts": {f"c{i}": 1 for i in range(16 if label_set == "omnifall16" else 5)},
    }
    data.update(extra or {})
    path = results / f"{run_id}.json"
    path.write_text(json.dumps(data))
    return path


def _both(results: Path) -> None:
    for seed, (f1, found) in enumerate([(0.88, 50), (0.90, 51), (0.91, 52)]):
        _run(results, PT, seed, f1, found)
    for seed, (f1, found) in enumerate([(0.86, 54), (0.87, 55), (0.87, 54)]):
        _run(results, TCN, seed, f1, found)


def test_loads_the_two_candidates_by_their_config(tmp_path: Path) -> None:
    results = tmp_path / "runs"
    _both(results)
    # runs that must be ignored: full pose, random start, another label set, another beta
    _run(results, PT, 0, 0.5, 10, pose="full")
    _run(results, PT, 0, 0.5, 10, pretrained=False)
    _run(results, PT, 0, 0.5, 10, label_set="groups5")
    _run(results, TCN, 0, 0.5, 10, beta=0.99)
    pt, tcn = choose.load_candidates(results)
    assert (pt.name, tcn.name) == (PT, TCN)
    assert pt.mean_f1 == pytest.approx((0.88 + 0.90 + 0.91) / 3)
    assert pt.f1_range == pytest.approx(0.03)
    assert pt.mean_detected == pytest.approx(51.0)
    assert pt.mean_missed == pytest.approx(6.0)
    assert pt.mean_false_alerts == pytest.approx(20.0)
    assert tcn.mean_detected == pytest.approx(54.3333, abs=1e-3)
    assert pt.parameters > tcn.parameters > 0
    assert pt.total_falls == FALLS


def test_a_candidate_needs_all_three_seeds(tmp_path: Path) -> None:
    results = tmp_path / "runs"
    _both(results)
    (results / f"{TCN}-omnifall16-b0.999-s2-lite-0.json").unlink()
    with pytest.raises(ValueError, match="per seed"):
        choose.load_candidates(results)


def test_reads_only_validation(tmp_path: Path) -> None:
    # a folder called test, a file called test.json and a result marked as the test split
    for path in (_run(tmp_path / "test", TCN, 0, 0.9, 50, name="a"),
                 _run(tmp_path, TCN, 0, 0.9, 50, name="test"),
                 _run(tmp_path, TCN, 0, 0.9, 50, name="b", extra={"split": "test"})):
        with pytest.raises(ValueError, match="validation"):
            choose.read_validation(path)
    fine = _run(tmp_path, TCN, 0, 0.9, 50, name="c", extra={"split": "val"})
    assert choose.read_validation(fine)["val"]["f1"] == 0.9
    # a test folder anywhere above is refused when the whole directory is loaded
    with pytest.raises(ValueError, match="validation"):
        choose.load_candidates(tmp_path / "test")


def test_markdown_shows_both_models_the_branch_and_the_pending_gate() -> None:
    pt, tcn = _cand(PT, 0.90, 51), _cand(TCN, 0.86, 54)
    text = choose.choice_markdown(pt, tcn, choose.choose(pt, tcn))
    assert "Provisional" in text
    assert "falls_found" in text
    assert "0.900" in text and "0.860" in text
    assert "51.0" in text and "54.0" in text
    assert "3,000,000" in text and "50,000" in text
    assert "not yet run" in text.lower()
    done = choose.choice_markdown(pt, tcn, choose.choose(pt, tcn, {PT: True, TCN: True}))
    assert "Provisional" not in done


def test_main_writes_the_report(tmp_path: Path) -> None:
    results, out = tmp_path / "runs", tmp_path / "f3b"
    _both(results)
    assert choose.main(["--results-dir", str(results), "--out-dir", str(out)]) == 0
    text = (out / "model-choice.md").read_text(encoding="utf-8")
    assert "tcn" in text.lower() and "falls_found" in text
