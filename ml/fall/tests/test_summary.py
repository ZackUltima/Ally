"""Tests for the F2 summary and its selection rules (F2 Task 7 Step 6, D-62)."""

import json
from pathlib import Path

import pytest

import ally_fall.summary as summary
from ally_fall.splits import CROSS_SUBJECT

BETA = 0.999
N_CLASSES = {"omnifall16": 16, "groups5": 5}


def _result(results: Path, model: str, label_set: str, seed: int, f1: float, *,
            pose: str = "lite", beta: float = BETA, pretrained: bool | None = None) -> None:
    """One run's JSON as `train.results_json` writes it (the fields the summary reads)."""
    pretrained = model == "stgcn_pt" if pretrained is None else pretrained
    start = "" if pretrained or model != "stgcn_pt" else "-rand"
    run_id = f"{model}-{label_set}-b{beta:g}-s{seed}{start}{'' if pose == 'lite' else '-' + pose}"
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{run_id}.json").write_text(json.dumps({
        "run_id": run_id,
        "config": {"run_id": run_id, "model": model, "label_set": label_set, "beta": beta,
                   "pose_model": pose, "seed": seed, "hidden": 8,
                   "checkpoint": "models/x.pth" if pretrained else ""},
        "val": {"f1": f1, "recall": f1 + 0.01, "precision": f1 - 0.01},
        "train_counts": {f"c{i}": 1 for i in range(N_CLASSES[label_set])},
    }))


def _seeds(results: Path, model: str, label_set: str, f1s: tuple[float, float, float],
           **kw: object) -> None:
    for seed, f1 in enumerate(f1s):
        _result(results, model, label_set, seed, f1, **kw)  # type: ignore[arg-type]


def _groups(results: Path) -> list[summary.Group]:
    return summary.group_results(summary.load_results(results))


def _grid(results: Path, pt: dict[str, tuple[float, float, float]]) -> None:
    """A full lite grid: from-scratch models at fixed scores, the pretrained one as given."""
    for label_set in ("omnifall16", "groups5"):
        _seeds(results, "tcn", label_set, (0.84, 0.85, 0.86))  # mean 0.85, range 0.02
        _seeds(results, "gru", label_set, (0.80, 0.81, 0.82))
        _seeds(results, "stgcn", label_set, (0.82, 0.83, 0.84))
        _seeds(results, "stgcn_pt", label_set, pt[label_set])


def _find(groups: list[summary.Group], model: str, label_set: str, *, pose: str = "lite",
          pretrained: bool | None = None) -> summary.Group:
    pretrained = model == "stgcn_pt" if pretrained is None else pretrained
    return next(g for g in groups if g.key == summary.Key(model, label_set, pose, BETA,
                                                          pretrained))


def test_groups_by_configuration_from_the_config_not_the_id(tmp_path: Path) -> None:
    _seeds(tmp_path, "stgcn_pt", "groups5", (0.8, 0.9, 1.0))
    _seeds(tmp_path, "stgcn_pt", "groups5", (0.5, 0.5, 0.5), pretrained=False)
    _seeds(tmp_path, "stgcn_pt", "groups5", (0.6, 0.6, 0.6), pose="full")
    (tmp_path / "beta-choice.json").write_text("{}")  # not a run: no config
    groups = _groups(tmp_path)
    assert len(groups) == 3
    pt = _find(groups, "stgcn_pt", "groups5")
    assert pt.complete and pt.mean == pytest.approx(0.9) and pt.spread == pytest.approx(0.2)
    assert _find(groups, "stgcn_pt", "groups5", pretrained=False).is_control
    assert not pt.is_control


def test_fewer_than_three_seeds_is_not_complete(tmp_path: Path) -> None:
    _result(tmp_path, "tcn", "groups5", 0, 0.8)
    _result(tmp_path, "tcn", "groups5", 1, 0.9)
    assert not _find(_groups(tmp_path), "tcn", "groups5").complete


def test_control_label_set_takes_the_pretrained_models_higher_mean(tmp_path: Path) -> None:
    _seeds(tmp_path, "stgcn_pt", "omnifall16", (0.90, 0.91, 0.92))
    _seeds(tmp_path, "stgcn_pt", "groups5", (0.80, 0.81, 0.82))
    assert summary.control_label_set(_groups(tmp_path), BETA) == "omnifall16"


def test_control_label_set_tie_takes_groups5(tmp_path: Path) -> None:
    for label_set in ("omnifall16", "groups5"):
        _seeds(tmp_path, "stgcn_pt", label_set, (0.80, 0.81, 0.82))
    assert summary.control_label_set(_groups(tmp_path), BETA) == "groups5"


def test_control_label_set_waits_for_all_seeds(tmp_path: Path) -> None:
    _seeds(tmp_path, "stgcn_pt", "omnifall16", (0.90, 0.91, 0.92))
    _result(tmp_path, "stgcn_pt", "groups5", 0, 0.99)
    assert summary.control_label_set(_groups(tmp_path), BETA) is None


@pytest.mark.parametrize(("pt", "beats"), [
    ((0.87, 0.88, 0.89), True),    # 0.88 - 0.85 = 0.03 > range 0.02
    ((0.85, 0.86, 0.87), False),   # 0.01: inside seed noise
    ((0.86, 0.87, 0.88), False),   # exactly the range: not more than it
])
def test_decision_6_beats_only_by_more_than_the_seed_range(
        tmp_path: Path, pt: tuple[float, float, float], beats: bool) -> None:
    _grid(tmp_path, {"omnifall16": pt, "groups5": pt})
    verdict = summary.decision_6(_groups(tmp_path), BETA, "omnifall16")
    assert verdict is not None
    assert verdict.beats is beats
    assert verdict.scratch.key.model == "tcn"  # the best from-scratch mean


def test_decision_6_compares_within_one_label_set(tmp_path: Path) -> None:
    _grid(tmp_path, {"omnifall16": (0.87, 0.88, 0.89), "groups5": (0.87, 0.88, 0.89)})
    _seeds(tmp_path, "gru", "groups5", (0.95, 0.95, 0.95))  # a stronger GRU, groups5 only
    groups = _groups(tmp_path)
    assert summary.decision_6(groups, BETA, "omnifall16").beats  # type: ignore[union-attr]
    verdict = summary.decision_6(groups, BETA, "groups5")
    assert verdict is not None and not verdict.beats and verdict.scratch.key.model == "gru"


def test_decision_6_waits_for_every_model(tmp_path: Path) -> None:
    _grid(tmp_path, {"omnifall16": (0.9, 0.9, 0.9), "groups5": (0.9, 0.9, 0.9)})
    (tmp_path / "gru-omnifall16-b0.999-s2.json").unlink()
    assert summary.decision_6(_groups(tmp_path), BETA, "omnifall16") is None
    assert summary.best_configuration(_groups(tmp_path), BETA) is None


def test_best_configuration_keeps_the_pretrained_model_only_where_it_beats(
        tmp_path: Path) -> None:
    # omnifall16: pretrained 0.86, a tie, so the TCN (0.85) is kept there; groups5: it beats
    _grid(tmp_path, {"omnifall16": (0.85, 0.86, 0.87), "groups5": (0.87, 0.88, 0.89)})
    best = summary.best_configuration(_groups(tmp_path), BETA)
    assert best is not None and best.key == summary.Key("stgcn_pt", "groups5", "lite", BETA, True)


def test_best_configuration_tie_with_seed_noise_keeps_from_scratch(tmp_path: Path) -> None:
    _grid(tmp_path, {"omnifall16": (0.85, 0.86, 0.87), "groups5": (0.85, 0.86, 0.87)})
    best = summary.best_configuration(_groups(tmp_path), BETA)
    assert best is not None and best.key.model == "tcn"


def test_control_is_never_the_best_configuration(tmp_path: Path) -> None:
    _grid(tmp_path, {"omnifall16": (0.8, 0.8, 0.8), "groups5": (0.8, 0.8, 0.8)})
    _seeds(tmp_path, "stgcn_pt", "groups5", (0.99, 0.99, 0.99), pretrained=False)
    best = summary.best_configuration(_groups(tmp_path), BETA)
    assert best is not None and not best.is_control


def test_best_configuration_equal_means_take_fewer_parameters(tmp_path: Path) -> None:
    _grid(tmp_path, {"omnifall16": (0.8, 0.8, 0.8), "groups5": (0.8, 0.8, 0.8)})
    for label_set in ("omnifall16", "groups5"):
        for model in ("tcn", "gru", "stgcn"):
            _seeds(tmp_path, model, label_set, (0.85, 0.85, 0.85))
    groups = _groups(tmp_path)
    best = summary.best_configuration(groups, BETA)
    smallest = min((g for g in groups if g.key.model != "stgcn_pt"), key=summary.param_count)
    assert best is not None and summary.param_count(best) == summary.param_count(smallest)
    assert best.key.label_set == "groups5"  # 5 output classes: fewer head weights than 16


def test_param_count_depends_on_the_label_set(tmp_path: Path) -> None:
    _grid(tmp_path, {"omnifall16": (0.8, 0.8, 0.8), "groups5": (0.8, 0.8, 0.8)})
    groups = _groups(tmp_path)
    for model in ("tcn", "stgcn_pt"):
        big = summary.param_count(_find(groups, model, "omnifall16"))
        small = summary.param_count(_find(groups, model, "groups5"))
        assert big > small > 0


@pytest.mark.parametrize(("full", "kept"), [
    ((0.88, 0.88, 0.88), "full"),   # 0.88 - 0.85 = 0.03 > lite range 0.02
    ((0.86, 0.86, 0.86), "lite"),
])
def test_pose_choice_keeps_lite_unless_full_beats_its_range(
        tmp_path: Path, full: tuple[float, float, float], kept: str) -> None:
    _seeds(tmp_path, "tcn", "groups5", (0.84, 0.85, 0.86))
    _seeds(tmp_path, "tcn", "groups5", full, pose="full")
    groups = _groups(tmp_path)
    assert summary.pose_choice(groups, _find(groups, "tcn", "groups5")) == kept


def test_pose_choice_waits_for_the_full_runs(tmp_path: Path) -> None:
    _seeds(tmp_path, "tcn", "groups5", (0.84, 0.85, 0.86))
    _result(tmp_path, "tcn", "groups5", 0, 0.99, pose="full")
    groups = _groups(tmp_path)
    assert summary.pose_choice(groups, _find(groups, "tcn", "groups5")) is None


def test_validation_people_come_from_the_split() -> None:
    people = summary.validation_people()
    assert len(people) == len(CROSS_SUBJECT)
    assert "UP-Fall subject 11" in people and "GMDCSA-24 subject 2" in people


def test_summary_before_any_result_marks_everything_pending(tmp_path: Path) -> None:
    text = summary.summary_markdown(tmp_path, None)
    assert "UP-Fall subject 11" in text
    assert text.count("pending") >= 5
    assert "test split" in text


def test_summary_with_every_result(tmp_path: Path) -> None:
    _grid(tmp_path, {"omnifall16": (0.85, 0.86, 0.87), "groups5": (0.87, 0.88, 0.89)})
    _seeds(tmp_path, "stgcn_pt", "groups5", (0.70, 0.71, 0.72), pretrained=False)
    _seeds(tmp_path, "stgcn_pt", "groups5", (0.86, 0.86, 0.86), pose="full")
    (tmp_path / "baseline-lite.json").write_text(json.dumps({
        "pose_model": "lite", "val": {"recall": 0.5, "precision": 0.4, "f1": 0.444,
                                      "specificity": 0.9},
        "intervals": {"f1": [0.3, 0.6]}}))
    path = summary.write_summary(tmp_path, BETA)
    assert path == tmp_path / "f2-summary.md"
    text = path.read_text(encoding="utf-8")
    assert "| lite | 0.500 | 0.400 | 0.444 (0.300–0.600) | 0.900 |" in text
    assert "| stgcn_pt | pretrained | groups5 | lite | 0.999 | 3 of 3 | 0.880 | 0.020 |" in text
    assert "random (control)" in text
    assert "omnifall16: tie, from-scratch kept" in text
    assert "groups5: pretrained beats" in text
    assert "**Control run label set (rule 1):** groups5" in text
    assert "pretrained 0.880 vs random start 0.710" in text
    assert "**Best configuration (rule 2):** stgcn_pt, pretrained, groups5" in text
    assert "**Pose model (rule 3):** lite" in text
    assert "pending" not in text.split("## Decisions")[1]
