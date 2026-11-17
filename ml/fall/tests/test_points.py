"""Tests for the `points = "coco17"` input switch (experiment D-71: 17 against 33 keypoints)."""

from pathlib import Path

import numpy as np
import pytest

from ally_fall.data import keep_coco17
from ally_fall.pretrained import coco17_from_mediapipe
from ally_fall.runs import write_config
from ally_fall.train import RunConfig, load_config, model_input

TICKS, POINTS = 30, 33


def _window() -> np.ndarray:
    """A (3, 30, 33) window where every point is visible at a distinct position."""
    rng = np.random.default_rng(1)
    w = rng.uniform(0.1, 0.9, size=(3, TICKS, POINTS)).astype(np.float32)
    w[2] = 1.0
    return w


def test_keeps_the_17_coco_points_and_zeroes_the_other_16() -> None:
    w = _window()
    out = keep_coco17(w)
    keep = list(coco17_from_mediapipe())
    drop = [p for p in range(POINTS) if p not in keep]
    assert len(keep) == 17 and len(drop) == 16
    assert out.shape == w.shape and out.dtype == np.float32
    assert np.array_equal(out[:, :, keep], w[:, :, keep])
    assert (out[:, :, drop] == 0.0).all()  # x, y and visibility: "missing", as the model knows it


def test_works_on_a_batch_and_does_not_change_its_input() -> None:
    batch = np.stack([_window(), _window()])
    before = batch.copy()
    out = keep_coco17(batch)
    assert out.shape == batch.shape
    assert np.array_equal(batch, before)
    assert np.array_equal(out[0], keep_coco17(batch[0]))


def test_rejects_a_wrong_shape() -> None:
    with pytest.raises(ValueError, match="33"):
        keep_coco17(np.zeros((3, TICKS, 17), dtype=np.float32))


def _cfg(**kw: object) -> RunConfig:
    return RunConfig(run_id="r", model="tcn", label_set="omnifall16", beta=0.999, **kw)  # type: ignore[arg-type]


def test_model_input_default_is_all_33_points() -> None:
    w = _window()
    assert _cfg().points == "all33"
    assert model_input(w, _cfg()) is w


def test_model_input_coco17_zeroes_the_extra_points() -> None:
    w = _window()
    assert np.array_equal(model_input(w, _cfg(points="coco17")), keep_coco17(w))


def test_coco17_and_hip_centred_together_keep_missing_points_zero() -> None:
    out = model_input(_window(), _cfg(points="coco17", hip_centred=True))
    drop = [p for p in range(POINTS) if p not in coco17_from_mediapipe()]
    assert (out[:, :, drop] == 0.0).all()


def test_load_config_reads_points_and_rejects_other_values(tmp_path: Path) -> None:
    base = 'run_id = "r"\nmodel = "tcn"\nlabel_set = "omnifall16"\nbeta = 0.999\n'
    ok = tmp_path / "ok.toml"
    ok.write_text(base + 'points = "coco17"\n', encoding="utf-8")
    assert load_config(ok).points == "coco17"
    bad = tmp_path / "bad.toml"
    bad.write_text(base + 'points = "all17"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="points"):
        load_config(bad)


def test_write_config_omits_points_at_its_default_and_round_trips_coco17(tmp_path: Path) -> None:
    plain = tmp_path / "plain.toml"
    write_config(_cfg(), plain)
    assert "points" not in plain.read_text(encoding="utf-8")
    coco = tmp_path / "coco.toml"
    write_config(_cfg(points="coco17"), coco)
    assert load_config(coco).points == "coco17"
