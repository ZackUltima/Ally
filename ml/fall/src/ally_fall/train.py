"""Train one fall model, or fit the threshold baseline, and score it on the validation people.

One TOML file per run (T4); the best epoch by validation fall-class F1 is kept (D-13, D-62).
F2 never opens the test split: it is used once, in F3 (technical plan §3).

Usage:
  python -m ally_fall.train --config configs/<run>.toml --data-root ~/ally-data
  python -m ally_fall.train --baseline --data-root ~/ally-data [--pose-model lite]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import get_args

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from ally_fall.augment import AugmentConfig, augment
from ally_fall.baseline import (
    RuleFeatures,
    Thresholds,
    fit_from_features,
    predict_from_features,
    rule_features,
)
from ally_fall.data import (
    BuiltSplit,
    load_split,
    person_of,
    window_tensor,
    window_video,
)
from ally_fall.labels import LabelSet, class_names, fall_classes, label_map
from ally_fall.metrics import (
    Binary,
    Events,
    binary,
    count_events,
    per_label_f1,
    person_bootstrap,
)
from ally_fall.models import ModelName, build_model
from ally_fall.omnifall import label_names
from ally_fall.ticks import TICK_S
from ally_fall.weights import class_weights

logger = logging.getLogger(__name__)

SPEED_GRID: tuple[float, ...] = tuple(np.round(np.arange(0.5, 5.01, 0.25), 2).tolist())
# The angle grid starts at 0: the first smoke fit (2026-10-20) chose 0.3, the old lowest value.
ANGLE_GRID: tuple[float, ...] = tuple(np.round(np.arange(0.0, 1.41, 0.1), 2).tolist())
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "runs"


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    model: ModelName
    label_set: LabelSet
    beta: float
    pose_model: str = "lite"
    hidden: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch: int = 256
    max_epochs: int = 40
    patience: int = 8
    seed: int = 0
    augment: AugmentConfig = field(default_factory=AugmentConfig)


def load_config(path: Path) -> RunConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    known = {f.name for f in fields(RunConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown}")
    if raw.get("model") not in get_args(ModelName):
        raise ValueError(f"{path}: model must be one of {get_args(ModelName)}")
    if raw.get("label_set") not in get_args(LabelSet):
        raise ValueError(f"{path}: label_set must be one of {get_args(LabelSet)}")
    try:
        aug = AugmentConfig(**raw.pop("augment", {}))
    except TypeError as error:
        raise ValueError(f"{path}: [augment] {error}") from error
    return RunConfig(**raw, augment=aug)


@dataclass(frozen=True)
class Scores:
    """Validation scores: fall class per window, per label, per event, person intervals."""

    val: Binary
    per_label: dict[str, float]
    events: Events
    intervals: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class RunResult(Scores):
    best_epoch: int = 0
    train_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class BaselineResult(Scores):
    thresholds: Thresholds = Thresholds(0.0, 0.0)


def _splits(data_root: Path, pose_model: str, label_set: LabelSet
            ) -> tuple[tuple[str, ...], BuiltSplit, BuiltSplit]:
    names = label_names(data_root / "omnifall")
    mapping = label_map(names, label_set)
    built = data_root / "built" / pose_model
    return names, load_split(built / "train.npz", mapping), load_split(built / "val.npz", mapping)


def score(pred: NDArray[np.int64] | None, fall_pred: NDArray[np.bool_], split: BuiltSplit,
          fall: tuple[int, ...], classes: tuple[str, ...]) -> Scores:
    """All validation measures from per-window predictions (class ids, and the fall class)."""
    true_fall = np.isin(split.y, fall)
    person = person_of(split)
    per_person = [binary(fall_pred[person == p], true_fall[person == p])
                  for p in np.unique(person)]
    video = window_video(split)
    events = Events(0, 0, 0)
    for v in np.unique(video):
        idx = np.flatnonzero(video == v)
        times = (split.window_end[idx] - split.video_start[v]) * TICK_S
        e = count_events(times, fall_pred[idx], split.y[idx], fall[0], fall[1])
        events = Events(events.detected + e.detected, events.missed + e.missed,
                        events.false_alerts + e.false_alerts)
    labels = per_label_f1(pred, split.y, len(classes)) if pred is not None else np.array([])
    return Scores(
        val=binary(fall_pred, true_fall),
        per_label={c: float(f) for c, f in zip(classes, labels, strict=False)},
        events=events,
        intervals={name: person_bootstrap(per_person, stat) for name, stat in (
            ("recall", lambda b: b.recall), ("precision", lambda b: b.precision),
            ("f1", lambda b: b.f1))},
    )


def _predict(model: nn.Module, split: BuiltSplit, device: str, chunk: int = 2048
             ) -> NDArray[np.int64]:
    model.eval()
    out: list[NDArray[np.int64]] = []
    with torch.no_grad():
        for start in range(0, len(split.window_end), chunk):
            idx = np.arange(start, min(start + chunk, len(split.window_end)))
            x = torch.from_numpy(window_tensor(split, idx)).to(device)
            out.append(model(x).argmax(1).cpu().numpy().astype(np.int64))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def train(cfg: RunConfig, data_root: Path, out: Path, device: str | None = None) -> RunResult:
    # cuBLAS reads this when CUDA first starts; it makes CUDA matrix products repeatable.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    rng = np.random.default_rng(cfg.seed)
    names, tr, va = _splits(data_root, cfg.pose_model, cfg.label_set)
    classes = class_names(cfg.label_set, names)
    fall = fall_classes(cfg.label_set, names)
    counts = np.bincount(tr.y, minlength=len(classes))
    weights = torch.from_numpy(class_weights(counts, cfg.beta)).to(device)
    model = build_model(cfg.model, len(classes), cfg.hidden).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    out.mkdir(parents=True, exist_ok=True)
    best_f1, best_epoch, waited = -1.0, 0, 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        order = rng.permutation(len(tr.window_end))
        for start in range(0, len(order), cfg.batch):
            idx = order[start:start + cfg.batch]
            windows = np.stack([augment(w, rng, cfg.augment) for w in window_tensor(tr, idx)])
            if len(idx) < 2:  # batch norm needs two windows
                continue
            optimiser.zero_grad()
            loss = loss_fn(model(torch.from_numpy(windows).to(device)),
                           torch.from_numpy(tr.y[idx]).to(device))
            loss.backward()
            optimiser.step()
        f1 = binary(np.isin(_predict(model, va, device), fall), np.isin(va.y, fall)).f1
        logger.info("%s epoch %d: validation fall-class F1 %.4f", cfg.run_id, epoch, f1)
        if f1 > best_f1:
            best_f1, best_epoch, waited = f1, epoch, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            waited += 1
            if waited >= cfg.patience:
                break
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    pred = _predict(model, va, device)
    scores = score(pred, np.isin(pred, fall), va, fall, classes)
    return RunResult(**scores.__dict__, best_epoch=best_epoch,
                     train_counts={c: int(n) for c, n in zip(classes, counts, strict=True)})


def _chunked_features(split: BuiltSplit, chunk: int) -> RuleFeatures:
    parts = [rule_features(window_tensor(split, np.arange(s, min(s + chunk,
                                                                 len(split.window_end)))))
             for s in range(0, len(split.window_end), chunk)]
    speed, angle, low = (np.concatenate(p) for p in zip(*parts, strict=True))
    return speed, angle, low


def run_baseline(data_root: Path, pose_model: str = "lite", chunk: int = 4096) -> BaselineResult:
    """Fit the threshold rule on the training windows; score it on validation."""
    names, tr, va = _splits(data_root, pose_model, "omnifall16")
    fall = fall_classes("omnifall16", names)
    t = fit_from_features(*_chunked_features(tr, chunk), np.isin(tr.y, fall), SPEED_GRID,
                          ANGLE_GRID)
    scores = score(None, predict_from_features(_chunked_features(va, chunk), t), va, fall, names)
    return BaselineResult(**scores.__dict__, thresholds=t)


def _measures(s: Scores) -> list[str]:
    b, ci = s.val, s.intervals
    return [
        "| measure (validation, fall class = fall ∪ fallen) | value | 95% interval by person |",
        "|---|---|---|",
        f"| recall | {b.recall:.3f} | {ci['recall'][0]:.3f}–{ci['recall'][1]:.3f} |",
        f"| precision | {b.precision:.3f} | {ci['precision'][0]:.3f}–{ci['precision'][1]:.3f} |",
        f"| F1 | {b.f1:.3f} | {ci['f1'][0]:.3f}–{ci['f1'][1]:.3f} |",
        f"| specificity | {b.specificity:.3f} | |",
        f"| TP / FP / FN / TN | {b.tp} / {b.fp} / {b.fn} / {b.tn} | |",
        f"| events detected / missed / false alerts | {s.events.detected} / {s.events.missed}"
        f" / {s.events.false_alerts} | |",
    ]


def results_markdown(cfg: RunConfig, r: RunResult) -> str:
    lines = [f"# Run {cfg.run_id}", "", "| setting | value |", "|---|---|"]
    lines += [f"| {f.name} | {getattr(cfg, f.name)} |" for f in fields(cfg)]
    lines += [f"| best epoch | {r.best_epoch} |", "", *_measures(r), "",
              "| label | training windows | validation F1 |", "|---|---|---|"]
    for c, f1 in r.per_label.items():
        n = r.train_counts.get(c, 0)
        lines.append(f"| {c} | {n if n else 'no training windows'} | {f1:.3f} |")
    return "\n".join(lines) + "\n"


def baseline_markdown(r: BaselineResult, pose_model: str) -> str:
    return "\n".join([
        "# Threshold baseline", "",
        f"Pose model {pose_model}; thresholds fitted on the training windows: hip drop speed"
        f" ≥ {r.thresholds.speed} torso lengths/s, torso angle ≥ {r.thresholds.angle} rad.", "",
        *_measures(r)]) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train a fall model or fit the baseline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path)
    group.add_argument("--baseline", action="store_true")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pose-model", default="lite", help="for --baseline")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    data_root = args.data_root.expanduser()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.baseline:
            result = run_baseline(data_root, args.pose_model)
            text, name = baseline_markdown(result, args.pose_model), f"baseline-{args.pose_model}"
        else:
            cfg = load_config(args.config)
            run = train(cfg, data_root, data_root / "runs" / cfg.run_id)
            text, name = results_markdown(cfg, run), cfg.run_id
    except (OSError, ValueError, KeyError) as error:
        logger.error("run failed: %s", error)
        return 1
    (args.results_dir / f"{name}.md").write_text(text, encoding="utf-8")
    logger.info("results written to %s", args.results_dir / f"{name}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
