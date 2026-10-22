import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

import ally_fall.train as train_module
from ally_fall.augment import AugmentConfig
from ally_fall.train import RunConfig, load_config, run_baseline, train

FIXTURES = Path(__file__).parent / "fixtures" / "omnifall"
WALK, FALL, FALLEN = 0, 1, 2


def _video(rng: np.random.Generator, ticks: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Walking, then a fast drop of the whole body (fall), then lying still (fallen)."""
    features = np.zeros((ticks, 33, 3), dtype=np.float32)
    base_y = np.full(ticks, 0.3)
    base_y[40:46] = np.linspace(0.3, 0.7, 6)
    base_y[46:] = 0.7
    features[:, :, 0] = rng.uniform(0.4, 0.6, size=(ticks, 33))
    features[:, :, 1] = base_y[:, None] + rng.uniform(-0.05, 0.05, size=(ticks, 33))
    features[:, :, 2] = 1.0
    labels = np.full(ticks, WALK, dtype=np.int64)
    labels[40:46] = FALL
    labels[46:] = FALLEN
    return features, labels


def _write_split(path: Path, videos: int, subject0: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    parts = [_video(rng) for _ in range(videos)]
    lengths = np.array([len(f) for f, _ in parts])
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    features = np.concatenate([f for f, _ in parts])
    labels = np.concatenate([lab for _, lab in parts])
    window_end = np.concatenate(
        [s + np.arange(29, n) for s, n in zip(starts, lengths, strict=True)])
    np.savez(path, features=features, labels=labels, video_start=starts, video_len=lengths,
             subject=subject0 + np.arange(videos), dataset=np.array(["GMDCSA24"] * videos),
             video_id=np.array([f"v{i}" for i in range(videos)]), window_end=window_end,
             y=labels[window_end])


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    (tmp_path / "omnifall" / "labels").mkdir(parents=True)
    shutil.copy(FIXTURES / "labels" / "label2id.csv", tmp_path / "omnifall" / "labels")
    built = tmp_path / "built" / "lite"
    built.mkdir(parents=True)
    _write_split(built / "train.npz", 4, 1, seed=1)
    _write_split(built / "val.npz", 2, 10, seed=2)
    (built / "test.npz").write_bytes(b"never to be opened in F2")
    return tmp_path


def _cfg(**changes: object) -> RunConfig:
    values: dict[str, object] = dict(run_id="t", model="tcn", label_set="omnifall16",
                                     beta=0.99, hidden=8, batch=32, max_epochs=2, patience=2)
    values.update(changes)
    return RunConfig(**values)  # type: ignore[arg-type]


def test_load_config_rejects_unknown_key(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text('run_id = "a"\nmodel = "tcn"\nlabel_set = "groups5"\nbeta = 0.99\nlr0 = 1\n')
    with pytest.raises(ValueError, match="lr0"):
        load_config(path)


def test_load_config_defaults_augment(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text('run_id = "a"\nmodel = "gru"\nlabel_set = "groups5"\nbeta = 0.999\n'
                    '[augment]\nedge_p = 0.1\n')
    cfg = load_config(path)
    assert cfg.augment == AugmentConfig(edge_p=0.1) and cfg.max_epochs == 40 and cfg.patience == 8
    assert (cfg.lr, cfg.weight_decay, cfg.batch, cfg.hidden, cfg.seed) == (1e-3, 1e-4, 256, 64, 0)


def test_train_never_opens_test_split(data_root: Path, tmp_path: Path) -> None:
    result = train(_cfg(), data_root, tmp_path / "out", device="cpu")
    assert 1 <= result.best_epoch <= 2
    assert (tmp_path / "out" / "best.pt").exists()


def test_validation_is_never_augmented(data_root: Path, tmp_path: Path,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    real = train_module.augment

    def counting(*args: object, **kwargs: object) -> np.ndarray:
        calls["n"] += 1
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(train_module, "augment", counting)
    train(_cfg(max_epochs=1), data_root, tmp_path / "out", device="cpu")
    train_windows = int(np.load(data_root / "built" / "lite" / "train.npz")["window_end"].size)
    assert calls["n"] == train_windows  # one call per training window, none for validation


def test_same_seed_same_result(data_root: Path, tmp_path: Path) -> None:
    a = train(_cfg(), data_root, tmp_path / "a", device="cpu")
    b = train(_cfg(), data_root, tmp_path / "b", device="cpu")
    assert a.val == b.val and a.best_epoch == b.best_epoch
    wa = torch.load(tmp_path / "a" / "best.pt")
    wb = torch.load(tmp_path / "b" / "best.pt")
    assert all(torch.equal(wa[k], wb[k]) for k in wa)


def test_results_table_written(data_root: Path, tmp_path: Path) -> None:
    result = train(_cfg(label_set="groups5"), data_root, tmp_path / "out", device="cpu")
    table = train_module.results_markdown(_cfg(label_set="groups5"), result)
    for text in ("| model | tcn |", "best epoch", "recall", "precision", "specificity",
                 "TP", "events", "lying", "no training windows"):  # lying has none here
        assert text in table, text


def test_run_baseline(data_root: Path) -> None:
    result = run_baseline(data_root, "lite", chunk=50)
    assert result.val.tp + result.val.fn > 0
    assert 0.0 <= result.val.f1 <= 1.0


def test_main_baseline_and_config(data_root: Path, tmp_path: Path) -> None:
    results = tmp_path / "results"
    assert train_module.main(["--baseline", "--data-root", str(data_root),
                              "--results-dir", str(results)]) == 0
    assert "Threshold baseline" in (results / "baseline-lite.md").read_text(encoding="utf-8")
    config = tmp_path / "c.toml"
    config.write_text('run_id = "m"\nmodel = "stgcn"\nlabel_set = "groups5"\nbeta = 0.9\n'
                      'hidden = 8\nbatch = 32\nmax_epochs = 1\n')
    assert train_module.main(["--config", str(config), "--data-root", str(data_root),
                              "--results-dir", str(results)]) == 0
    assert "| model | stgcn |" in (results / "m.md").read_text(encoding="utf-8")
    bad = tmp_path / "bad.toml"
    bad.write_text('run_id = "x"\nmodel = "lstm"\nlabel_set = "groups5"\nbeta = 0.9\n')
    assert train_module.main(["--config", str(bad), "--data-root", str(data_root),
                              "--results-dir", str(results)]) == 1
