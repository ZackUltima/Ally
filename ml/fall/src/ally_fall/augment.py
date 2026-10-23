"""Training augmentation for 30-tick keypoint windows (D-55 item 8, D-62).

A window is (3, 30, 33): x, y, visibility per tick and point, x and y in [0, 1] of the
picture. A missing point is exactly (0, 0, 0) and stays so after every change; a point
moved outside the picture becomes missing, as ticks.mark_missing does for real frames.
Rotation and noise are F3 ablation switches only, off by default. Training uses
`augment_batch` (a whole batch in array operations, 2026-10-22); the one-window functions
stay as its reference, and a test checks that both give the same result for the same draw.
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


@dataclass(frozen=True)
class BatchDraw:
    """Every random choice for a batch of B windows, drawn at once (the batch form of augment)."""

    mirror: NDArray[np.bool_]  # (B,)
    rx: NDArray[np.float64]  # (B,)
    ry: NDArray[np.float64]  # (B,)
    angle: NDArray[np.float64] | None  # (B,), None when rotation is off
    noise: NDArray[np.float32] | None  # (B, 2, T, V), None when noise is off
    drop: NDArray[np.bool_]  # (B, T, V)
    edge: NDArray[np.bool_]  # (B,)
    edge_side: NDArray[np.int64]  # (B,), index into EDGES
    edge_at: NDArray[np.float64]  # (B,)


def draw_batch(
    n: int, shape: tuple[int, ...], rng: np.random.Generator, cfg: AugmentConfig
) -> BatchDraw:
    """Draw the changes for n windows of (3, *shape); same distributions as augment."""
    return BatchDraw(
        mirror=rng.random(n) < cfg.mirror_p,
        rx=rng.uniform(-cfg.scale, cfg.scale, n),
        ry=rng.uniform(-cfg.scale, cfg.scale, n),
        angle=rng.uniform(-cfg.rotate, cfg.rotate, n) if cfg.rotate > 0 else None,
        noise=(rng.normal(0.0, cfg.noise_sd, (n, 2, *shape)).astype(np.float32)
               if cfg.noise_sd > 0 else None),
        drop=rng.random((n, *shape)) < cfg.dropout_p,
        edge=rng.random(n) < cfg.edge_p,
        edge_side=rng.integers(4, size=n),
        edge_at=rng.uniform(0.0, cfg.edge_max, n),
    )


def _settle_batch(b: Window, missing: NDArray[np.bool_]) -> Window:
    """_settle for a batch (B, 3, T, V) in one pass: a point is zeroed if it is given as
    missing, lies outside the picture (or is NaN, as ticks.mark_missing), or has visibility 0."""
    x, y = b[:, 0], b[:, 1]
    with np.errstate(invalid="ignore"):
        outside = ~((x >= 0.0) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0))  # NaN fails every test
    gone = missing | outside | (b[:, 2] == 0)
    return np.where(gone[:, None], np.float32(0.0), b).astype(np.float32, copy=False)


def _where(pick: NDArray[np.bool_], new: Window, old: Window) -> Window:
    return np.where(pick[:, None, None, None], new, old).astype(np.float32)


def apply_batch(b: Window, d: BatchDraw, aspect: float = 4 / 3) -> Window:
    """Apply a draw to a batch (B, 3, T, V), step for step as augment does to one window."""
    flipped = b[:, :, :, _SWAP].copy()
    missing = flipped[:, 2] == 0
    flipped[:, 0] = 1.0 - flipped[:, 0]
    out = _where(d.mirror, _settle_batch(flipped, missing), b)

    seen = out[:, 2][:, :, HIPS] > 0  # scale about each window's mean visible hip centre
    count = seen.sum(axis=(1, 2))
    has_hips = count > 0
    safe = np.maximum(count, 1)
    cx = (out[:, 0][:, :, HIPS] * seen).sum(axis=(1, 2)) / safe
    cy = (out[:, 1][:, :, HIPS] * seen).sum(axis=(1, 2)) / safe
    scaled = out.copy()
    scaled[:, 0] = cx[:, None, None] + (out[:, 0] - cx[:, None, None]) * (1.0 + d.rx[:, None, None])
    scaled[:, 1] = cy[:, None, None] + (out[:, 1] - cy[:, None, None]) * (1.0 + d.ry[:, None, None])
    out = _where(has_hips, _settle_batch(scaled, out[:, 2] == 0), out)

    if d.angle is not None:
        cos, sin = np.cos(d.angle)[:, None, None], np.sin(d.angle)[:, None, None]
        px, py = (out[:, 0] - 0.5) * aspect, out[:, 1] - 0.5
        turned = out.copy()
        turned[:, 0] = (px * cos - py * sin) / aspect + 0.5
        turned[:, 1] = px * sin + py * cos + 0.5
        out = _settle_batch(turned, out[:, 2] == 0)
    if d.noise is not None:
        moved = out.copy()
        moved[:, 0:2] = out[:, 0:2] + d.noise
        out = _settle_batch(moved, out[:, 2] == 0)
    out = _settle_batch(out, (out[:, 2] == 0) | d.drop)

    x, y = out[:, 0], out[:, 1]
    at = d.edge_at[:, None, None]
    beyond = np.stack([x < at, x > 1.0 - at, y < at, y > 1.0 - at])  # in EDGES order
    beyond = np.take_along_axis(beyond, d.edge_side[None, :, None, None], axis=0)[0]
    return _where(d.edge, _settle_batch(out, (out[:, 2] == 0) | beyond), out)


def augment_batch(
    b: Window, rng: np.random.Generator, cfg: AugmentConfig, aspect: float = 4 / 3
) -> Window:
    """augment for a whole batch (B, 3, T, V) at once: one draw, array operations only."""
    return apply_batch(b, draw_batch(len(b), b.shape[2:], rng, cfg), aspect)


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
