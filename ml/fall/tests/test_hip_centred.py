"""Tests for the hip-centred input switch (F3b Task 2, D-55 item 8)."""

from pathlib import Path

import numpy as np
import pytest

from ally_fall.augment import HIPS
from ally_fall.data import hip_centred
from ally_fall.train import RunConfig, load_config, model_input

TICKS, POINTS = 30, 33


def _window(hip: tuple[float, float] = (0.4, 0.6)) -> np.ndarray:
    """A (3, 30, 33) window: both hips visible at `hip` in every tick, one other point visible."""
    w = np.zeros((3, TICKS, POINTS), dtype=np.float32)
    for point in HIPS:
        w[0, :, point], w[1, :, point], w[2, :, point] = hip[0], hip[1], 1.0
    w[0, :, 0], w[1, :, 0], w[2, :, 0] = 0.9, 0.5, 1.0
    return w


def test_hip_centred_subtracts_the_mean_visible_hip() -> None:
    w = _window((0.4, 0.6))
    w[0, :, HIPS[1]] = 0.6  # the two hips at x 0.4 and 0.6: mean 0.5
    out = hip_centred(w)
    assert out.dtype == np.float32 and out.shape == w.shape
    assert np.allclose(out[0, :, 0], 0.9 - 0.5)
    assert np.allclose(out[1, :, 0], 0.5 - 0.6)
    assert np.allclose(out[0, :, HIPS[0]], -0.1) and np.allclose(out[0, :, HIPS[1]], 0.1)
    assert np.array_equal(out[2], w[2])  # visibility is untouched


def test_the_mean_uses_visible_hip_ticks_only() -> None:
    w = _window((0.4, 0.6))
    w[:, :10, HIPS[0]] = 0.0  # left hip missing in the first 10 ticks
    w[:, :10, HIPS[1]] = 0.0  # and the right one too: those ticks do not count
    w[0, 10:, HIPS[0]], w[0, 10:, HIPS[1]] = 0.2, 0.4  # visible ticks: mean x is 0.3
    out = hip_centred(w)
    assert np.allclose(out[0, 10:, 0], 0.9 - 0.3)


def test_missing_points_stay_exactly_zero() -> None:
    w = _window()
    w[:, :, 5] = 0.0  # point 5 is missing everywhere
    w[:, 3, 0] = 0.0  # and the nose is missing in tick 3
    out = hip_centred(w)
    assert (out[:, :, 5] == 0.0).all()
    assert (out[:, 3, 0] == 0.0).all()


def test_window_with_no_visible_hip_is_left_unchanged() -> None:
    w = _window()
    w[:, :, list(HIPS)] = 0.0
    assert np.array_equal(hip_centred(w), w)


def test_does_not_change_its_input_and_batches_match_single_windows() -> None:
    a, b = _window((0.3, 0.4)), _window((0.7, 0.8))
    batch = np.stack([a, b])
    before = batch.copy()
    out = hip_centred(batch)
    assert np.array_equal(batch, before)
    assert out.shape == batch.shape
    assert np.allclose(out[0], hip_centred(a), atol=1e-6)  # same up to float rounding
    assert np.allclose(out[1], hip_centred(b), atol=1e-6)


def test_model_input_is_unchanged_unless_the_switch_is_on() -> None:
    batch = np.stack([_window((0.3, 0.4)), _window((0.7, 0.8))])
    off = RunConfig(run_id="a", model="tcn", label_set="groups5", beta=0.99)
    on = RunConfig(run_id="b", model="tcn", label_set="groups5", beta=0.99, hip_centred=True)
    assert model_input(batch, off) is batch
    assert np.array_equal(model_input(batch, on), hip_centred(batch))


def test_default_is_off_and_old_configs_still_load(tmp_path: Path) -> None:
    path = tmp_path / "old.toml"
    path.write_text('run_id = "a"\nmodel = "tcn"\nlabel_set = "groups5"\nbeta = 0.999\nseed = 1\n')
    cfg = load_config(path)
    assert cfg.hip_centred is False
    assert cfg == RunConfig(run_id="a", model="tcn", label_set="groups5", beta=0.999, seed=1)


def test_the_switch_can_be_turned_on_in_a_config(tmp_path: Path) -> None:
    path = tmp_path / "new.toml"
    path.write_text('run_id = "a"\nmodel = "tcn"\nlabel_set = "groups5"\nbeta = 0.999\n'
                    "hip_centred = true\n")
    assert load_config(path).hip_centred is True


@pytest.mark.parametrize("shape", [(30, 33), (2, 2, 3, 30, 33)])
def test_a_wrong_shape_is_refused(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="shape"):
        hip_centred(np.zeros(shape, dtype=np.float32))
