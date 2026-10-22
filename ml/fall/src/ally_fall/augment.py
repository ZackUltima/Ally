"""Training augmentation for 30-tick keypoint windows (D-55 item 8, D-62).

A window is (3, 30, 33): x, y, visibility per tick and point, x and y in [0, 1] of the
picture. A missing point is exactly (0, 0, 0) and stays so after every change; a point
moved outside the picture becomes missing, as ticks.mark_missing does for real frames.
Rotation and noise are F3 ablation switches only, off by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ally_fall.ticks import mark_missing

Window = NDArray[np.float32]
Edge = Literal["left", "right", "top", "bottom"]
EDGES: tuple[Edge, ...] = ("left", "right", "top", "bottom")

# MediaPipe Pose left/right partners (the nose, point 0, has none).
LEFT_RIGHT: tuple[tuple[int, int], ...] = (
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
    (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
)
_SWAP: NDArray[np.int64] = np.arange(33, dtype=np.int64)
for _left, _right in LEFT_RIGHT:
    _SWAP[_left], _SWAP[_right] = _right, _left
HIPS: tuple[int, int] = (23, 24)


@dataclass(frozen=True)
class AugmentConfig:
    mirror_p: float = 0.5
    scale: float = 0.2
    dropout_p: float = 0.05
    edge_p: float = 0.3
    edge_max: float = 0.3
    rotate: float = 0.0
    noise_sd: float = 0.0


def _settle(w: Window, missing: NDArray[np.bool_]) -> Window:
    """Zero the given missing points, then mark points outside the picture missing."""
    xy = np.moveaxis(w[0:2], 0, -1)
    xy = np.where(missing[..., None], np.float32(0.0), xy)
    vis = np.where(missing, np.float32(0.0), w[2])
    xy, vis = mark_missing(xy.astype(np.float32), vis.astype(np.float32))
    xy = np.where((vis == 0)[..., None], np.float32(0.0), xy)
    return np.concatenate([np.moveaxis(xy, -1, 0), vis[None]], axis=0).astype(np.float32)


def mirror(w: Window) -> Window:
    """x becomes 1 - x and every left point swaps with its right partner."""
    out = w[:, :, _SWAP].copy()
    missing = out[2] == 0
    out[0] = 1.0 - out[0]
    return _settle(out, missing)


def scale(w: Window, rx: float, ry: float) -> Window:
    """Stretch x by 1 + rx and y by 1 + ry about the window's mean visible hip centre."""
    seen = w[2][:, HIPS] > 0
    if not seen.any():
        return w.copy()
    cx = float(w[0][:, HIPS][seen].mean())
    cy = float(w[1][:, HIPS][seen].mean())
    out = w.copy()
    out[0] = cx + (w[0] - cx) * (1.0 + rx)
    out[1] = cy + (w[1] - cy) * (1.0 + ry)
    return _settle(out, w[2] == 0)


def dropout(w: Window, rng: np.random.Generator, p: float) -> Window:
    """Mark each (tick, point) missing with probability p."""
    return _settle(w.copy(), (w[2] == 0) | (rng.random(w.shape[1:]) < p))


def edge_cut(w: Window, edge: Edge, at: float) -> Window:
    """Mark missing, for the whole window, every point beyond a line `at` from one edge."""
    x, y = w[0], w[1]
    beyond = {"left": x < at, "right": x > 1.0 - at, "top": y < at, "bottom": y > 1.0 - at}[edge]
    return _settle(w.copy(), (w[2] == 0) | beyond)


def rotate(w: Window, angle: float, aspect: float) -> Window:
    """Rotate about the picture centre in true proportions (aspect = width / height)."""
    out = w.copy()
    px, py = (w[0] - 0.5) * aspect, w[1] - 0.5
    cos, sin = np.cos(angle), np.sin(angle)
    out[0] = (px * cos - py * sin) / aspect + 0.5
    out[1] = px * sin + py * cos + 0.5
    return _settle(out, w[2] == 0)


def noise(w: Window, rng: np.random.Generator, sd: float) -> Window:
    """Add Gaussian noise of standard deviation sd to the x and y of visible points."""
    out = w.copy()
    out[0:2] = w[0:2] + rng.normal(0.0, sd, size=w[0:2].shape).astype(np.float32)
    return _settle(out, w[2] == 0)


def augment(
    w: Window, rng: np.random.Generator, cfg: AugmentConfig, aspect: float = 4 / 3
) -> Window:
    """One random draw of every enabled change, in a fixed order."""
    out = mirror(w) if rng.random() < cfg.mirror_p else w.copy()
    out = scale(out, float(rng.uniform(-cfg.scale, cfg.scale)),
                float(rng.uniform(-cfg.scale, cfg.scale)))
    if cfg.rotate > 0:
        out = rotate(out, float(rng.uniform(-cfg.rotate, cfg.rotate)), aspect)
    if cfg.noise_sd > 0:
        out = noise(out, rng, cfg.noise_sd)
    out = dropout(out, rng, cfg.dropout_p)
    if rng.random() < cfg.edge_p:
        out = edge_cut(out, EDGES[int(rng.integers(4))], float(rng.uniform(0.0, cfg.edge_max)))
    return out
