import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import ally_fall.train as train_module
from ally_fall.ablate import SETTINGS, apply
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
    real = train_module.augment_batch

    def counting(windows: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        calls["n"] += len(windows)
        return real(windows, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(train_module, "augment_batch", counting)
    train(_cfg(max_epochs=1), data_root, tmp_path / "out", device="cpu")
    train_windows = int(np.load(data_root / "built" / "lite" / "train.npz")["window_end"].size)
    assert calls["n"] == train_windows  # every training window once, no validation window


def test_load_config_accepts_checkpoint_keys(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text('run_id = "p"\nmodel = "stgcn_pt"\nlabel_set = "groups5"\nbeta = 0.99\n'
                    'checkpoint = "models/stgcn_pt/x.pth"\nbackbone_lr = 1e-4\n')
    cfg = load_config(path)
    assert (cfg.checkpoint, cfg.backbone_lr) == ("models/stgcn_pt/x.pth", 1e-4)
    assert (_cfg().checkpoint, _cfg().backbone_lr) == ("", None)


def test_backbone_lr_param_groups() -> None:
    model = nn.Sequential()
    model.add_module("body", nn.Linear(3, 4))
    model.add_module("fc", nn.Linear(4, 2))
    two = train_module.param_groups(model, lr=1e-3, backbone_lr=1e-4)
    assert [g["lr"] for g in two] == [1e-4, 1e-3]
    assert [len(g["params"]) for g in two] == [2, 2]
    assert two[1]["params"] == list(model.fc.parameters())
    one = train_module.param_groups(model, lr=1e-3, backbone_lr=None)
    assert len(one) == 1 and one[0]["lr"] == 1e-3 and len(one[0]["params"]) == 4


def test_train_pretrained_from_checkpoint(data_root: Path, tmp_path: Path) -> None:
    from ally_fall.pretrained import PretrainedSTGCN

    torch.manual_seed(3)
    source = PretrainedSTGCN(60)
    sd = {("cls_head." if k.startswith("fc.") else "backbone.") + k: v
          for k, v in source.state_dict().items()}
    (data_root / "models").mkdir()
    torch.save({"state_dict": sd}, data_root / "models" / "ckpt.pth")
    cfg = _cfg(model="stgcn_pt", checkpoint="models/ckpt.pth", backbone_lr=1e-4, max_epochs=1)
    result = train(cfg, data_root, tmp_path / "pt", device="cpu")
    assert 0.0 <= result.val.f1 <= 1.0 and result.best_epoch == 1
    saved = torch.load(tmp_path / "pt" / "best.pt")
    assert saved["fc.weight"].shape[0] == 16  # the new head, omnifall16


def test_resume_after_interruption_matches_uninterrupted(
        data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(max_epochs=3, patience=3)
    full = train(cfg, data_root, tmp_path / "full", device="cpu")
    real = train_module._predict
    calls = {"n": 0}

    def killed_in_epoch_2(*args: object, **kwargs: object) -> np.ndarray:
        calls["n"] += 1
        if calls["n"] == 2:  # epoch 2 trained, killed before its state was saved
            raise KeyboardInterrupt
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(train_module, "_predict", killed_in_epoch_2)
    with pytest.raises(KeyboardInterrupt):
        train(cfg, data_root, tmp_path / "cut", device="cpu")
    monkeypatch.setattr(train_module, "_predict", real)
    trained = {"n": 0}
    real_aug = train_module.augment_batch

    def counting(windows: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        trained["n"] += len(windows)
        return real_aug(windows, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(train_module, "augment_batch", counting)
    resumed = train(cfg, data_root, tmp_path / "cut", device="cpu")
    per_epoch = int(np.load(data_root / "built" / "lite" / "train.npz")["window_end"].size)
    assert trained["n"] == 2 * per_epoch  # epochs 2 and 3 only: epoch 1 is not redone
    assert resumed.val == full.val and resumed.best_epoch == full.best_epoch
    wa, wb = torch.load(tmp_path / "full" / "best.pt"), torch.load(tmp_path / "cut" / "best.pt")
    assert all(torch.equal(wa[k], wb[k]) for k in wa)


def test_finished_run_is_not_trained_again(data_root: Path, tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    first = train(_cfg(), data_root, tmp_path / "out", device="cpu")
    monkeypatch.setattr(train_module, "augment_batch", lambda *a, **k: pytest.fail("trained"))
    again = train(_cfg(), data_root, tmp_path / "out", device="cpu")
    assert again.val == first.val and again.best_epoch == first.best_epoch


def test_resume_refuses_a_different_config(data_root: Path, tmp_path: Path) -> None:
    train(_cfg(), data_root, tmp_path / "out", device="cpu")
    with pytest.raises(ValueError, match="different config"):
        train(_cfg(lr=1e-2), data_root, tmp_path / "out", device="cpu")


def test_mixed_precision_only_on_cuda() -> None:
    assert train_module.use_amp(_cfg(), "cuda") and train_module.use_amp(_cfg(), "cuda:0")
    assert not train_module.use_amp(_cfg(), "cpu")
    assert not train_module.use_amp(_cfg(amp=False), "cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU (laptop only)")
def test_trains_with_mixed_precision_on_gpu(data_root: Path, tmp_path: Path) -> None:
    result = train(_cfg(max_epochs=1), data_root, tmp_path / "gpu", device="cuda")
    assert 0.0 <= result.val.f1 <= 1.0


def test_queue_runs_each_config_once(data_root: Path, tmp_path: Path) -> None:
    queue, results = tmp_path / "queue", tmp_path / "results"
    queue.mkdir()
    for run_id in ("a", "b"):
        (queue / f"{run_id}.toml").write_text(
            f'run_id = "{run_id}"\nmodel = "tcn"\nlabel_set = "groups5"\nbeta = 0.9\n'
            'hidden = 8\nbatch = 32\nmax_epochs = 1\n')
    results.mkdir()
    (results / "a.md").write_text("done before", encoding="utf-8")
    args = ["--queue", str(queue), "--data-root", str(data_root), "--results-dir", str(results)]
    assert train_module.main(args) == 0
    assert (results / "a.md").read_text(encoding="utf-8") == "done before"  # skipped
    assert "| run_id | b |" in (results / "b.md").read_text(encoding="utf-8")
    saved = json.loads((results / "b.json").read_text(encoding="utf-8"))
    assert saved["run_id"] == "b" and saved["config"]["model"] == "tcn"
    assert 0.0 <= saved["val"]["f1"] <= 1.0 and saved["best_epoch"] == 1
    assert {"recall", "precision", "specificity", "tp", "fp", "fn", "tn"} <= set(saved["val"])
    assert set(saved["intervals"]) == {"recall", "precision", "f1"} and "events" in saved
    (queue / "c.toml").write_text(
        'run_id = "c"\nmodel = "lstm"\nlabel_set = "groups5"\nbeta = 0.9\n')
    assert train_module.main(args) == 1  # a bad config is reported; the others are not redone


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
    data = json.loads((results / "baseline-lite.json").read_text(encoding="utf-8"))
    assert data["pose_model"] == "lite" and 0.0 <= data["val"]["f1"] <= 1.0
    assert set(data["val"]) == {"tp", "fp", "fn", "tn", "recall", "precision", "f1",
                                "specificity"}
    assert len(data["intervals"]["f1"]) == 2 and "speed" in data["thresholds"]
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


def test_trains_with_hip_centred_input(data_root: Path, tmp_path: Path) -> None:
    result = train(_cfg(hip_centred=True), data_root, tmp_path / "out", device="cpu")
    assert result.best_epoch >= 1 and (tmp_path / "out" / "best.pt").exists()
    plain = train(_cfg(), data_root, tmp_path / "plain", device="cpu")
    wa = torch.load(tmp_path / "out" / "best.pt")
    wb = torch.load(tmp_path / "plain" / "best.pt")
    assert not all(torch.equal(wa[k], wb[k]) for k in wa), "centring must change what is learned"
    assert plain.best_epoch >= 1


def test_trains_on_shorter_windows_and_every_ablation_setting(data_root: Path,
                                                              tmp_path: Path) -> None:
    for setting in SETTINGS:  # one short run of each of the nine settings: none may crash
        cfg = apply(_cfg(max_epochs=1, patience=1), setting)
        result = train(cfg, data_root, tmp_path / setting.name, device="cpu")
        assert result.best_epoch >= 1, setting.name
