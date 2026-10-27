"""Tests for the live fall scorer (F3c Task 3, D-69)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from ally_fall.data import hip_centred
from ally_fall.keypoints import POINTS
from ally_fall.live.scorer import Scorer
from ally_fall.models import build_model
from ally_fall.omnifall import label_names

NAMES = (
    "walk", "fall", "fallen", "sit_down", "sitting", "lie_down", "lying", "stand_up",
    "standing", "other", "kneel_down", "kneeling", "squat_down", "squatting",
    "crawl", "jump",
)


def _omnifall(root: Path) -> Path:
    labels = root / "omnifall" / "labels"
    labels.mkdir(parents=True)
    rows = "\n".join(f"{name},{i}" for i, name in enumerate(NAMES))
    (labels / "label2id.csv").write_text("label,id\n" + rows + "\n", encoding="utf-8")
    return root / "omnifall"


def _config(path: Path, *, label_set: str = "omnifall16", extra: str = "") -> Path:
    path.write_text(
        'run_id = "t"\nmodel = "tcn"\n'
        f'label_set = "{label_set}"\nbeta = 0.999\npose_model = "lite"\nhidden = 8\n{extra}',
        encoding="utf-8",
    )
    return path


def _weights(path: Path, *, classes: int = 16, ticks: int = 30, seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = build_model("tcn", classes, hidden=8, ticks=ticks)
    model.eval()
    torch.save(model.state_dict(), path)
    return model


def _windows(count: int, ticks: int = 30, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-0.5, 1.0, (count, 3, ticks, POINTS)).astype(
        np.float32
    )


@pytest.fixture
def scorer_files(tmp_path: Path) -> tuple[Path, Path, Path, torch.nn.Module]:
    model = _weights(tmp_path / "best.pt")
    return _config(tmp_path / "run.toml"), tmp_path / "best.pt", _omnifall(tmp_path), model


def test_the_score_is_the_fall_and_fallen_probability(
    scorer_files: tuple[Path, Path, Path, torch.nn.Module],
) -> None:
    config, weights, omnifall, model = scorer_files
    scorer = Scorer.load(config, weights, omnifall)
    windows = _windows(5)
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(windows)), dim=1).numpy()
    names = label_names(omnifall)
    expected = probs[:, names.index("fall")] + probs[:, names.index("fallen")]
    assert np.allclose(scorer.score_batch(windows), expected, atol=1e-6)
    assert scorer.score(windows[2]) == pytest.approx(float(expected[2]), abs=1e-6)
    assert 0.0 <= scorer.score(windows[0]) <= 1.0


def test_hip_centring_follows_the_config(tmp_path: Path) -> None:
    model = _weights(tmp_path / "best.pt")
    omnifall = _omnifall(tmp_path)
    config = _config(tmp_path / "run.toml", extra="hip_centred = true\n")
    scorer = Scorer.load(config, tmp_path / "best.pt", omnifall)
    window = _windows(1)[0]
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(hip_centred(window)[None])), dim=1).numpy()[0]
    names = label_names(omnifall)
    assert scorer.score(window) == pytest.approx(
        float(probs[names.index("fall")] + probs[names.index("fallen")]), abs=1e-6
    )


def test_window_length_follows_the_config(tmp_path: Path) -> None:
    _weights(tmp_path / "best.pt", ticks=21)
    omnifall = _omnifall(tmp_path)
    config = _config(tmp_path / "run.toml", extra="window_ticks = 21\n")
    scorer = Scorer.load(config, tmp_path / "best.pt", omnifall)
    assert scorer.window_ticks == 21
    assert 0.0 <= scorer.score(_windows(1, ticks=21)[0]) <= 1.0
    with pytest.raises(ValueError, match="window"):
        scorer.score(_windows(1, ticks=30)[0])


def test_a_wrong_window_shape_is_refused(
    scorer_files: tuple[Path, Path, Path, torch.nn.Module],
) -> None:
    scorer = Scorer.load(*scorer_files[:3])
    with pytest.raises(ValueError, match="window"):
        scorer.score(np.zeros((2, 30, POINTS), dtype=np.float32))
    with pytest.raises(ValueError, match="window"):
        scorer.score_batch(np.zeros((3, 30, POINTS), dtype=np.float32))


def test_a_checkpoint_for_another_label_set_is_refused(tmp_path: Path) -> None:
    _weights(tmp_path / "best.pt", classes=5)  # a groups5 model
    omnifall = _omnifall(tmp_path)
    config = _config(tmp_path / "run.toml")  # says omnifall16
    with pytest.raises(ValueError, match="does not match"):
        Scorer.load(config, tmp_path / "best.pt", omnifall)


def test_a_missing_checkpoint_is_refused(tmp_path: Path) -> None:
    omnifall = _omnifall(tmp_path)
    config = _config(tmp_path / "run.toml")
    with pytest.raises(FileNotFoundError):
        Scorer.load(config, tmp_path / "nope.pt", omnifall)


def test_a_groups5_model_uses_its_own_fall_classes(tmp_path: Path) -> None:
    model = _weights(tmp_path / "best.pt", classes=5)
    omnifall = _omnifall(tmp_path)
    config = _config(tmp_path / "run.toml", label_set="groups5")
    scorer = Scorer.load(config, tmp_path / "best.pt", omnifall)
    window = _windows(1)[0]
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(window[None])), dim=1).numpy()[0]
    fall_and_fallen = float(probs[0] + probs[1])  # classes 0 and 1 of groups5
    assert scorer.score(window) == pytest.approx(fall_and_fallen, abs=1e-6)


def test_top_class_is_the_most_likely_class_and_matches_the_fall_mask(
    scorer_files: tuple[Path, Path, Path, torch.nn.Module],
) -> None:
    config, weights, omnifall, model = scorer_files
    scorer = Scorer.load(config, weights, omnifall)
    windows = _windows(8, seed=5)
    with torch.no_grad():
        expected = model(torch.from_numpy(windows)).argmax(1).numpy()
    top = scorer.top_class_batch(windows)
    assert np.array_equal(top, expected)
    names = label_names(omnifall)
    fall_ids = [names.index("fall"), names.index("fallen")]
    assert np.array_equal(scorer.top_is_fall_batch(windows), np.isin(expected, fall_ids))
