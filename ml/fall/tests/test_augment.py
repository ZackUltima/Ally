import math
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from ally_fall.augment import (
    LEFT_RIGHT,
    AugmentConfig,
    augment,
    dropout,
    edge_cut,
    mirror,
    noise,
    rotate,
    scale,
)

Window = NDArray[np.float32]


def _window(seed: int = 0, missing: float = 0.2) -> Window:
    """A (3, 30, 33) window: visible points in [0.2, 0.8], some points missing (0, 0, 0)."""
    rng = np.random.default_rng(seed)
    w = np.zeros((3, 30, 33), dtype=np.float32)
    w[0:2] = rng.uniform(0.2, 0.8, size=(2, 30, 33))
    w[2] = rng.uniform(0.5, 1.0, size=(30, 33))
    hidden = rng.random((30, 33)) < missing
    w[:, hidden] = 0.0
    w[:, :, 23:25] = np.array([0.5, 0.6, 0.9], dtype=np.float32)[:, None, None]  # hips visible
    return w


def _missing(w: Window) -> NDArray[np.bool_]:
    return w[2] == 0


def test_mirror_twice_is_identity() -> None:
    w = _window()
    np.testing.assert_allclose(mirror(mirror(w)), w, atol=1e-6)


def test_mirror_swaps_exactly_left_right() -> None:
    w = _window(missing=0.0)
    m = mirror(w)
    assert dict(LEFT_RIGHT)[11] == 12 and len(LEFT_RIGHT) == 16
    np.testing.assert_allclose(m[0, :, 11], 1.0 - w[0, :, 12], atol=1e-6)
    np.testing.assert_allclose(m[1, :, 11], w[1, :, 12])
    np.testing.assert_allclose(m[0, :, 0], 1.0 - w[0, :, 0], atol=1e-6)  # the nose stays point 0


CHANGES: dict[str, Callable[[Window], Window]] = {
    "mirror": mirror,
    "scale": lambda w: scale(w, 0.2, -0.2),
    "dropout": lambda w: dropout(w, np.random.default_rng(1), 0.05),
    "edge_cut": lambda w: edge_cut(w, "bottom", 0.2),
    "rotate": lambda w: rotate(w, 0.1, 16 / 9),
    "noise": lambda w: noise(w, np.random.default_rng(1), 0.01),
}


@pytest.mark.parametrize("name", list(CHANGES))
def test_missing_stays_missing_after_each_change(name: str) -> None:
    w = _window()
    out = CHANGES[name](w)
    was_missing = _missing(w)
    if name == "mirror":  # mirrored points move to their partner's slot
        was_missing = _missing(mirror(w))
    assert np.all(out[:, was_missing] == 0.0)


def _hip_centre(w: Window) -> NDArray[np.float32]:
    hips = w[:, :, 23:25]
    seen = hips[2] > 0
    return np.array([hips[0][seen].mean(), hips[1][seen].mean()], dtype=np.float32)


def test_scale_keeps_hip_centre() -> None:
    w = _window()
    np.testing.assert_allclose(_hip_centre(scale(w, 0.15, -0.1)), _hip_centre(w), atol=1e-6)


def test_scale_without_visible_hips_is_unchanged() -> None:
    w = _window()
    w[:, :, 23:25] = 0.0
    np.testing.assert_array_equal(scale(w, 0.2, 0.2), w)


def test_scaled_point_outside_image_becomes_missing() -> None:
    w = _window(missing=0.0)
    w[0:2, :, 0] = 0.98  # near the corner, far from the hip centre (0.5, 0.6)
    out = scale(w, 0.2, 0.2)
    assert np.all(out[:, :, 0] == 0.0)
    assert np.all(out[2, :, 23] > 0)


def test_edge_cut_hides_only_points_beyond_line() -> None:
    w = _window()
    out = edge_cut(w, "bottom", 0.2)
    beyond = (w[1] > 0.8) & ~_missing(w)
    assert np.all(out[:, beyond] == 0.0)
    np.testing.assert_array_equal(out[:, ~beyond], w[:, ~beyond])


def test_rotation_respects_aspect() -> None:
    w = np.zeros((3, 30, 33), dtype=np.float32)
    w[:, :, 0] = np.array([0.45, 0.5, 1.0], dtype=np.float32)[:, None]
    w[:, :, 1] = np.array([0.55, 0.5, 1.0], dtype=np.float32)[:, None]  # 0.1 wide on a 16:9 frame
    out = rotate(w, math.pi / 2, 16 / 9)
    dx = (out[0, 0, 1] - out[0, 0, 0]) * 16
    dy = (out[1, 0, 1] - out[1, 0, 0]) * 9
    assert abs(dx) < 1e-5 and abs(abs(dy) - 1.6) < 1e-5


def test_dropout_rate() -> None:
    cfg = AugmentConfig(mirror_p=0.0, scale=0.0, dropout_p=0.05, edge_p=0.0)
    rng = np.random.default_rng(3)
    w = _window(missing=0.0)
    new = [float(_missing(augment(w, rng, cfg)).mean()) for _ in range(200)]
    assert abs(float(np.mean(new)) - 0.05) < 0.005


def test_same_seed_same_result() -> None:
    w = _window()
    a = augment(w, np.random.default_rng(7), AugmentConfig())
    b = augment(w, np.random.default_rng(7), AugmentConfig())
    np.testing.assert_array_equal(a, b)


def test_defaults_have_no_rotation_or_noise() -> None:
    cfg = AugmentConfig()
    assert cfg.rotate == 0.0 and cfg.noise_sd == 0.0
    assert (cfg.mirror_p, cfg.scale, cfg.dropout_p, cfg.edge_p, cfg.edge_max) == (
        0.5, 0.2, 0.05, 0.3, 0.3)
