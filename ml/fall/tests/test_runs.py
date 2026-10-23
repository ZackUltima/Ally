"""Tests for the F2 run plan and the overnight sequence (F2 Task 7 Step 5, D-62)."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

import ally_fall.runs as runs
import ally_fall.summary as summary
from ally_fall.train import RunConfig, load_config


def _fake_result(results: Path, run_id: str, f1: float) -> None:
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{run_id}.json").write_text(json.dumps({"run_id": run_id, "val": {"f1": f1}}))


def test_beta_configs_are_the_tcn_on_omnifall16_seed_0() -> None:
    cfgs = runs.beta_configs()
    assert [c.beta for c in cfgs] == [0.99, 0.999, 0.9999]
    assert {(c.model, c.label_set, c.seed, c.pose_model) for c in cfgs} == {
        ("tcn", "omnifall16", 0, "lite")}
    assert [c.run_id for c in cfgs] == ["tcn-omnifall16-b0.99-s0", "tcn-omnifall16-b0.999-s0",
                                        "tcn-omnifall16-b0.9999-s0"]


def test_grid_is_the_approved_24_runs() -> None:
    cfgs = runs.grid_configs(0.999)
    assert len(cfgs) == 24 and len({c.run_id for c in cfgs}) == 24
    for model in ("tcn", "gru", "stgcn", "stgcn_pt"):
        mine = [c for c in cfgs if c.model == model]
        assert {(c.label_set, c.seed) for c in mine} == {
            (s, seed) for s in ("omnifall16", "groups5") for seed in (0, 1, 2)}
    assert all(c.beta == 0.999 and c.pose_model == "lite" and c.amp for c in cfgs)
    pretrained = [c for c in cfgs if c.model == "stgcn_pt"]
    assert all(c.checkpoint == runs.CHECKPOINT and c.backbone_lr == 1e-4 for c in pretrained)
    others = [c for c in cfgs if c.model != "stgcn_pt"]
    assert all(c.checkpoint == "" and c.backbone_lr is None for c in others)
    # the chosen beta's TCN run is the same run as its beta run, so the queue never repeats it
    assert runs.beta_configs()[1].run_id in {c.run_id for c in cfgs}


def test_config_file_round_trip(tmp_path: Path) -> None:
    for cfg in [*runs.beta_configs(), *runs.grid_configs(0.99)]:
        path = tmp_path / f"{cfg.run_id}.toml"
        runs.write_config(cfg, path)
        assert load_config(path) == cfg


def test_pick_beta_takes_the_best_validation_f1(tmp_path: Path) -> None:
    for beta, f1 in ((0.99, 0.80), (0.999, 0.86), (0.9999, 0.84)):
        _fake_result(tmp_path, f"tcn-omnifall16-b{beta}-s0", f1)
    assert runs.pick_beta(tmp_path) == 0.999


def test_pick_beta_tie_takes_the_smaller_beta(tmp_path: Path) -> None:
    for beta in (0.99, 0.999, 0.9999):
        _fake_result(tmp_path, f"tcn-omnifall16-b{beta}-s0", 0.85)
    assert runs.pick_beta(tmp_path) == 0.99


def test_pick_beta_needs_all_three(tmp_path: Path) -> None:
    _fake_result(tmp_path, "tcn-omnifall16-b0.99-s0", 0.8)
    with pytest.raises(FileNotFoundError, match="b0.999"):
        runs.pick_beta(tmp_path)


def test_overnight_runs_baseline_beta_pick_then_grid(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    results, configs = tmp_path / "results", tmp_path / "configs"
    order: list[str] = []
    scores = {0.99: 0.80, 0.999: 0.81, 0.9999: 0.86}

    def fake_queue(folder: Path, data_root: Path, results_dir: Path) -> int:
        order.append(folder.name)
        for path in sorted(folder.glob("*.toml")):
            cfg = load_config(path)
            _fake_result(results_dir, cfg.run_id, scores.get(cfg.beta, 0.5))
        return 0

    def fake_baseline(data_root: Path, results_dir: Path) -> int:
        order.append("baseline")
        return 0

    monkeypatch.setattr(runs, "run_queue", fake_queue)
    monkeypatch.setattr(runs, "run_baseline_once", fake_baseline)
    assert runs.overnight(tmp_path / "data", results, configs) == 0
    assert order == ["baseline", "f2-beta", "f2-grid"]
    grid = sorted((configs / "f2-grid").glob("*.toml"))
    assert len(grid) == 24 and all(load_config(p).beta == 0.9999 for p in grid)
    choice = (results / "beta-choice.md").read_text(encoding="utf-8")
    assert "0.9999" in choice and "0.860" in choice


def test_overnight_stops_before_the_grid_if_a_beta_run_failed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runs, "run_queue", lambda folder, data_root, results_dir: 1)
    monkeypatch.setattr(runs, "run_baseline_once", lambda data_root, results_dir: 0)
    assert runs.overnight(tmp_path / "data", tmp_path / "results", tmp_path / "configs") == 1
    assert not (tmp_path / "configs" / "f2-grid").exists()


def test_main_overnight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[Path, Path, Path]] = []
    monkeypatch.setattr(runs, "overnight", lambda d, r, c: seen.append((d, r, c)) or 0)
    assert runs.main(["overnight", "--data-root", str(tmp_path), "--results-dir",
                      str(tmp_path / "r"), "--configs-dir", str(tmp_path / "c")]) == 0
    assert seen == [(tmp_path, tmp_path / "r", tmp_path / "c")]


def test_control_and_full_ids_never_match_a_grid_id() -> None:
    grid = {c.run_id for c in runs.grid_configs(0.999)}
    control = runs.control_configs(0.999, "groups5")
    full = runs.full_configs(summary.Key("stgcn_pt", "groups5", "lite", 0.999, True))
    assert [c.run_id for c in control] == [f"stgcn_pt-groups5-b0.999-s{s}-rand" for s in (0, 1, 2)]
    assert [c.run_id for c in full] == [f"stgcn_pt-groups5-b0.999-s{s}-full" for s in (0, 1, 2)]
    assert not grid & {c.run_id for c in [*control, *full]}


def test_control_is_the_pretrained_network_from_a_random_start_at_one_rate() -> None:
    for cfg in runs.control_configs(0.99, "omnifall16"):
        assert (cfg.model, cfg.label_set, cfg.beta, cfg.pose_model) == (
            "stgcn_pt", "omnifall16", 0.99, "lite")
        assert cfg.checkpoint == "" and cfg.backbone_lr is None and cfg.lr == 1e-3


def test_full_configs_copy_the_configuration_on_full_poses(tmp_path: Path) -> None:
    pretrained = runs.full_configs(summary.Key("stgcn_pt", "omnifall16", "lite", 0.99, True))
    assert all(c.pose_model == "full" and c.checkpoint == runs.CHECKPOINT
               and c.backbone_lr == runs.BACKBONE_LR for c in pretrained)
    tcn = runs.full_configs(summary.Key("tcn", "groups5", "lite", 0.99, False))
    assert all(c.model == "tcn" and c.checkpoint == "" and c.pose_model == "full" for c in tcn)
    for cfg in [*pretrained, *tcn, *runs.control_configs(0.99, "groups5")]:
        path = tmp_path / f"{cfg.run_id}.toml"
        runs.write_config(cfg, path)
        assert load_config(path) == cfg


def _full_result(results: Path, cfg: RunConfig, f1: float) -> None:
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{cfg.run_id}.json").write_text(json.dumps({
        "run_id": cfg.run_id, "config": asdict(cfg),
        "val": {"f1": f1, "recall": f1, "precision": f1},
        "train_counts": {f"c{i}": 1 for i in range(16 if cfg.label_set == "omnifall16" else 5)},
    }))


def test_after_runs_control_then_full_baseline_and_full_runs(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results, configs = tmp_path / "results", tmp_path / "configs"
    for cfg in runs.beta_configs():
        _full_result(results, cfg, 0.9 if cfg.beta == 0.999 else 0.5)
    for cfg in runs.grid_configs(0.999):
        if not (results / f"{cfg.run_id}.json").exists():
            _full_result(results, cfg, 0.95 if cfg.model == "gru" else 0.5)
    order: list[str] = []

    def fake_queue(folder: Path, data_root: Path, results_dir: Path) -> int:
        order.append(folder.name)
        for path in sorted(folder.glob("*.toml")):
            _full_result(results_dir, load_config(path), 0.6)
        return 0

    def fake_baseline(data_root: Path, results_dir: Path, pose_model: str = "lite") -> int:
        order.append(f"baseline-{pose_model}")
        return 0

    monkeypatch.setattr(runs, "run_queue", fake_queue)
    monkeypatch.setattr(runs, "run_baseline_once", fake_baseline)
    assert runs.after(tmp_path / "data", results, configs) == 0
    assert order == ["f2-control", "baseline-full", "f2-full"]
    control = [load_config(p) for p in sorted((configs / "f2-control").glob("*.toml"))]
    assert len(control) == 3 and {c.label_set for c in control} == {"groups5"}  # tie: groups5
    full = [load_config(p) for p in sorted((configs / "f2-full").glob("*.toml"))]
    assert len(full) == 3 and {(c.model, c.pose_model) for c in full} == {("gru", "full")}
    text = (results / "f2-summary.md").read_text(encoding="utf-8")
    assert "**Pose model (rule 3):** lite" in text


def test_after_stops_when_the_grid_is_not_finished(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    results = tmp_path / "results"
    for cfg in runs.beta_configs():
        _full_result(results, cfg, 0.9)
    monkeypatch.setattr(runs, "run_queue", lambda folder, data_root, results_dir: 0)
    assert runs.after(tmp_path / "data", results, tmp_path / "configs") == 1
    assert not (tmp_path / "configs" / "f2-control").exists()


def test_overnight_ends_with_the_summary(tmp_path: Path,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_queue(folder: Path, data_root: Path, results_dir: Path) -> int:
        for path in folder.glob("*.toml"):
            _fake_result(results_dir, load_config(path).run_id, 0.8)
        return 0

    monkeypatch.setattr(runs, "run_queue", fake_queue)
    monkeypatch.setattr(runs, "run_baseline_once", lambda data_root, results_dir: 0)
    assert runs.overnight(tmp_path / "data", tmp_path / "results", tmp_path / "configs") == 0
    assert "pending" in (tmp_path / "results" / "f2-summary.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("step", ["after", "summary"])
def test_main_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step: str) -> None:
    seen: list[str] = []
    monkeypatch.setattr(runs, "after", lambda d, r, c: seen.append("after") or 0)
    assert runs.main([step, "--data-root", str(tmp_path), "--results-dir",
                      str(tmp_path / "r"), "--configs-dir", str(tmp_path / "c")]) == 0
    if step == "after":
        assert seen == ["after"]
    else:
        assert seen == [] and "pending" in (tmp_path / "r" / "f2-summary.md").read_text(
            encoding="utf-8")
