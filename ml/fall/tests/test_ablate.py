"""Tests for the F3b ablation configs and summary (F3b Task 3, D-55 item 8, D-68 decision 2)."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import ally_fall.ablate as ablate
from ally_fall.data import BuiltSplit, window_tensor
from ally_fall.runs import write_config
from ally_fall.train import RunConfig, load_config

SEEDS = (0, 1, 2)


def _base(seed: int) -> RunConfig:
    return RunConfig(run_id=f"tcn-omnifall16-b0.999-s{seed}", model="tcn", label_set="omnifall16",
                     beta=0.999, seed=seed)


def _result(directory: Path, cfg: RunConfig, f1: float, found: int) -> None:
    """One run's JSON with the fields `train.results_json` writes that the summary reads."""
    directory.mkdir(parents=True, exist_ok=True)
    config = asdict(cfg)
    for key in ("hip_centred", "window_ticks"):  # F2 results were written before these existed
        if directory.name == "runs":
            config.pop(key)
    (directory / f"{cfg.run_id}.json").write_text(json.dumps({
        "run_id": cfg.run_id, "config": config, "best_epoch": 3,
        "val": {"f1": f1, "recall": f1, "precision": f1},
        "events": {"detected": found, "missed": 57 - found, "false_alerts": 20},
        "train_counts": {f"c{i}": 1 for i in range(16)},
    }))


def _flat(cfg: RunConfig) -> dict[str, Any]:
    return ablate.flat(cfg)


def _bases(tmp_path: Path) -> Path:
    runs = tmp_path / "runs"
    for seed, f1 in zip(SEEDS, (0.864, 0.868, 0.873), strict=True):
        _result(runs, _base(seed), f1, 54)
    return runs


def test_writes_nine_settings_times_three_seeds(tmp_path: Path) -> None:
    configs = ablate.ablation_configs([_base(s) for s in SEEDS])
    assert len(ablate.SETTINGS) == 9 and len(configs) == 27
    assert len({c.run_id for c in configs}) == 27
    assert sorted({c.seed for c in configs}) == [0, 1, 2]
    paths = ablate.write_configs(configs, tmp_path / "f3b-ablate")
    assert len(paths) == 27
    assert [load_config(p) for p in paths] == configs  # every config survives the round trip


def test_each_config_differs_from_the_base_in_one_setting() -> None:
    bases = {s: _base(s) for s in SEEDS}
    for cfg in ablate.ablation_configs(list(bases.values())):
        base = _flat(bases[cfg.seed])
        changed = {k: v for k, v in _flat(cfg).items() if k != "run_id" and v != base[k]}
        setting = ablate.setting_of(cfg, bases[cfg.seed])
        assert setting is not None, cfg.run_id
        assert changed == setting.changes, cfg.run_id


def test_the_nine_settings_are_the_ones_d55_names() -> None:
    changes = {s.name: s.changes for s in ablate.SETTINGS}
    assert changes["hip_centred"] == {"hip_centred": True}
    assert changes["window_21"] == {"window_ticks": 21}
    assert changes["no_augment"] == {"augment.mirror_p": 0.0, "augment.scale": 0.0,
                                     "augment.dropout_p": 0.0, "augment.edge_p": 0.0}
    for name, key in (("no_mirror", "mirror_p"), ("no_scale", "scale"),
                      ("no_dropout", "dropout_p"), ("no_edge", "edge_p")):
        assert changes[name] == {f"augment.{key}": 0.0}
    assert changes["rotation"] == {"augment.rotate": 0.1}
    assert changes["noise"] == {"augment.noise_sd": 0.01}


def test_default_configs_are_written_exactly_as_before(tmp_path: Path) -> None:
    write_config(_base(0), tmp_path / "a.toml")
    text = (tmp_path / "a.toml").read_text(encoding="utf-8")
    assert "hip_centred" not in text and "window_ticks" not in text and "[augment]" not in text


def test_the_base_configs_are_read_from_the_f2_results(tmp_path: Path) -> None:
    runs = _bases(tmp_path)
    _result(runs, RunConfig(run_id="gru", model="gru", label_set="omnifall16", beta=0.999), 0.9, 50)
    other_beta = RunConfig(run_id="tcn-b99", model="tcn", label_set="omnifall16", beta=0.99)
    _result(runs, other_beta, 0.9, 50)
    bases = ablate.base_configs(runs)
    assert bases == [_base(s) for s in SEEDS]


def _ablation_results(tmp_path: Path) -> tuple[Path, Path]:
    runs, out = _bases(tmp_path), tmp_path / "f3b-runs"
    bases = ablate.base_configs(runs)
    scores = {  # setting: (per-seed F1, falls found)
        "hip_centred": ((0.890, 0.895, 0.900), 55),    # +0.027, range 0.010: better
        "window_21": ((0.830, 0.835, 0.840), 52),      # -0.033: worse
        "no_mirror": ((0.866, 0.870, 0.875), 54),      # +0.001, range 0.009: no clear difference
        "rotation": ((0.860, 0.870, 0.880), 54),       # +0.001, range 0.020: no clear difference
    }
    for cfg in ablate.ablation_configs(bases):
        setting = ablate.setting_of(cfg, bases[cfg.seed])
        assert setting is not None
        if setting.name in scores:
            f1s, found = scores[setting.name]
            _result(out, cfg, f1s[cfg.seed], found)
    # one setting with only two of its seeds finished
    partial = next(c for c in ablate.ablation_configs(bases)
                   if (ablate.setting_of(c, bases[c.seed]) or ablate.SETTINGS[0]).name == "noise")
    _result(out, partial, 0.87, 54)
    return runs, out


def test_summary_orders_settings_and_marks_overlap(tmp_path: Path) -> None:
    runs, out = _ablation_results(tmp_path)
    rows = ablate.summarise(runs, out)
    names = [r.setting for r in rows if r.complete]
    assert names == ["hip_centred", "no_mirror", "rotation", "window_21"]  # best gap first
    verdicts = {r.setting: r.verdict for r in rows}
    assert verdicts["hip_centred"] == "better"
    assert verdicts["window_21"] == "worse"
    assert verdicts["no_mirror"] == "no clear difference"
    assert verdicts["rotation"] == "no clear difference"
    assert verdicts["noise"] == "incomplete"
    text = ablate.summary_markdown(runs, out)
    assert "hip_centred" in text and "BEATS the F2 setting" in text and "ask the PO" in text
    assert "1 of 3" in text  # the unfinished setting is shown as such


def test_summary_without_any_better_setting_says_so(tmp_path: Path) -> None:
    runs = _bases(tmp_path)
    bases = ablate.base_configs(runs)
    for cfg in ablate.ablation_configs(bases):
        _result(tmp_path / "f3b-runs", cfg, 0.86, 54)
    text = ablate.summary_markdown(runs, tmp_path / "f3b-runs")
    assert "BEATS" not in text


def test_summary_never_reads_the_test_split(tmp_path: Path) -> None:
    runs = _bases(tmp_path)
    with pytest.raises(ValueError, match="validation"):
        ablate.summarise(runs, tmp_path / "test" / "runs")


def test_main_writes_configs_and_the_summary(tmp_path: Path) -> None:
    runs, out = _ablation_results(tmp_path)
    configs, results = tmp_path / "f3b-ablate", tmp_path / "f3b"
    assert ablate.main(["configs", "--results-dir", str(runs), "--configs-dir", str(configs)]) == 0
    assert len(list(configs.glob("*.toml"))) == 27
    assert ablate.main(["summary", "--results-dir", str(runs), "--runs-dir", str(out),
                        "--out-dir", str(results)]) == 0
    assert "hip_centred" in (results / "ablations.md").read_text(encoding="utf-8")


def test_ablations_run_on_windows_of_other_lengths() -> None:
    n_ticks = 40
    split = BuiltSplit(
        features=np.arange(n_ticks * 33 * 3, dtype=np.float32).reshape(n_ticks, 33, 3),
        video_start=np.array([0]), video_len=np.array([n_ticks]), subject=np.array([1]),
        dataset=np.array(["d"]), window_end=np.array([35]), y=np.array([0]))
    assert window_tensor(split, np.array([0])).shape == (1, 3, 30, 33)
    short = window_tensor(split, np.array([0]), length=21)
    assert short.shape == (1, 3, 21, 33)
    assert np.array_equal(short[0], window_tensor(split, np.array([0]))[0][:, -21:])
