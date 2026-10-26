"""Train one fall model, or fit the threshold baseline, and score it on the validation people.

One TOML file per run (T4); the best epoch by validation fall-class F1 is kept (D-13, D-62).
F2 never opens the test split: it is used once, in F3 (technical plan §3).

Usage:
  python -m ally_fall.train --config configs/<run>.toml --data-root ~/ally-data
  python -m ally_fall.train --baseline --data-root ~/ally-data [--pose-model lite]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, get_args

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from ally_fall.augment import AugmentConfig, augment_batch
from ally_fall.baseline import (
    RuleFeatures,
    Thresholds,
    fit_from_features,
    predict_from_features,
    rule_features,
)
from ally_fall.data import (
    BuiltSplit,
    hip_centred,
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
    checkpoint: str = ""  # path under the data root; empty = random start (stgcn_pt only)
    backbone_lr: float | None = None  # None = one learning rate for every layer (D-62 decision 5)
    amp: bool = True  # mixed precision on a CUDA GPU (speed-up, PO 2026-10-22); ignored on CPU
    hip_centred: bool = False  # F3b ablation: move each window to its mean visible hip (D-55)
    window_ticks: int = 30  # F3b ablation: the last N ticks of each built window (21 = 2.1 s)


def model_input(windows: NDArray[np.float32], cfg: RunConfig) -> NDArray[np.float32]:
    """The windows as the model sees them: unchanged, or hip-centred when the config says so."""
    return hip_centred(windows) if cfg.hip_centred else windows


def param_groups(model: nn.Module, lr: float, backbone_lr: float | None) -> list[dict[str, Any]]:
    """One AdamW group, or two: every layer but the head at `backbone_lr`, the head (`fc`) at
    `lr` (fine-tuning a pretrained model, D-62 decision 5)."""
    if backbone_lr is None:
        return [{"params": list(model.parameters()), "lr": lr}]
    head = [p for n, p in model.named_parameters() if n.startswith("fc.")]
    body = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    return [{"params": body, "lr": backbone_lr}, {"params": head, "lr": lr}]


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


def _predict(model: nn.Module, split: BuiltSplit, device: str, cfg: RunConfig,
             chunk: int = 2048) -> NDArray[np.int64]:
    model.eval()
    out: list[NDArray[np.int64]] = []
    with torch.no_grad():
        for start in range(0, len(split.window_end), chunk):
            idx = np.arange(start, min(start + chunk, len(split.window_end)))
            windows = model_input(window_tensor(split, idx, cfg.window_ticks), cfg)
            x = torch.from_numpy(windows).to(device)
            out.append(model(x).argmax(1).cpu().numpy().astype(np.int64))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def use_amp(cfg: RunConfig, device: str) -> bool:
    """Mixed precision (float16 on the GPU, float32 weights) only where it helps: CUDA."""
    return cfg.amp and device.startswith("cuda")


@dataclass(frozen=True)
class _Progress:
    epoch: int = 0  # last finished epoch
    best_f1: float = -1.0
    best_epoch: int = 0
    waited: int = 0
    stopped: bool = False  # early stopping already reached


def _save_state(path: Path, cfg: RunConfig, progress: _Progress, model: nn.Module,
                optimiser: torch.optim.Optimizer, scaler: torch.amp.GradScaler,
                rng: np.random.Generator, device: str) -> None:
    """Everything needed to continue after the last finished epoch, written atomically."""
    state: dict[str, Any] = {
        "config": asdict(cfg), "progress": asdict(progress), "model": model.state_dict(),
        "optimiser": optimiser.state_dict(), "scaler": scaler.state_dict(),
        "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
    }
    if device.startswith("cuda"):
        state["cuda_rng"] = torch.cuda.get_rng_state(torch.device(device))
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _resume(path: Path, cfg: RunConfig, model: nn.Module, optimiser: torch.optim.Optimizer,
            scaler: torch.amp.GradScaler, rng: np.random.Generator, device: str) -> _Progress:
    """Restore a run interrupted after a finished epoch, so it ends exactly as if it never was."""
    state = torch.load(path, map_location=device, weights_only=True)
    if state["config"] != asdict(cfg):
        raise ValueError(f"{path} was saved for a different config; move it to start afresh")
    model.load_state_dict(state["model"])
    optimiser.load_state_dict(state["optimiser"])
    scaler.load_state_dict(state["scaler"])
    rng.bit_generator.state = state["rng"]
    torch.set_rng_state(state["torch_rng"].cpu())
    if "cuda_rng" in state and device.startswith("cuda"):
        torch.cuda.set_rng_state(state["cuda_rng"].cpu(), torch.device(device))
    progress = _Progress(**state["progress"])
    logger.info("%s: resuming after epoch %d", cfg.run_id, progress.epoch)
    return progress


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
    checkpoint = data_root / cfg.checkpoint if cfg.checkpoint else None
    model = build_model(cfg.model, len(classes), cfg.hidden, checkpoint, cfg.window_ticks)
    model = model.to(device)
    optimiser = torch.optim.AdamW(param_groups(model, cfg.lr, cfg.backbone_lr), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    amp = use_amp(cfg, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    out.mkdir(parents=True, exist_ok=True)
    progress = _Progress()
    state_file = out / "last.pt"
    if state_file.exists():
        progress = _resume(state_file, cfg, model, optimiser, scaler, rng, device)
    for epoch in range(progress.epoch + 1, cfg.max_epochs + 1):
        if progress.stopped:
            break
        model.train()
        order = rng.permutation(len(tr.window_end))
        for start in range(0, len(order), cfg.batch):
            idx = order[start:start + cfg.batch]
            windows = augment_batch(window_tensor(tr, idx, cfg.window_ticks), rng, cfg.augment)
            windows = model_input(windows, cfg)
            if len(idx) < 2:  # batch norm needs two windows
                continue
            optimiser.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                loss = loss_fn(model(torch.from_numpy(windows).to(device)),
                               torch.from_numpy(tr.y[idx]).to(device))
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
        f1 = binary(np.isin(_predict(model, va, device, cfg), fall), np.isin(va.y, fall)).f1
        logger.info("%s epoch %d: validation fall-class F1 %.4f", cfg.run_id, epoch, f1)
        if f1 > progress.best_f1:
            progress = replace(progress, best_f1=f1, best_epoch=epoch, waited=0)
            torch.save(model.state_dict(), out / "best.pt")
        else:
            progress = replace(progress, waited=progress.waited + 1)
        progress = replace(progress, epoch=epoch, stopped=progress.waited >= cfg.patience)
        _save_state(state_file, cfg, progress, model, optimiser, scaler, rng, device)
    best_epoch = progress.best_epoch
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    pred = _predict(model, va, device, cfg)
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


def results_json(cfg: RunConfig, r: RunResult) -> dict[str, Any]:
    """The same results as the table, as data (read by the beta pick and the F2 summary)."""
    return {
        "run_id": cfg.run_id, "config": asdict(cfg), "best_epoch": r.best_epoch,
        **_scores_json(r), "per_label": r.per_label, "train_counts": r.train_counts,
    }


def _scores_json(s: Scores) -> dict[str, Any]:
    b = s.val
    return {"val": {"tp": b.tp, "fp": b.fp, "fn": b.fn, "tn": b.tn, "recall": b.recall,
                    "precision": b.precision, "f1": b.f1, "specificity": b.specificity},
            "intervals": {k: list(v) for k, v in s.intervals.items()},
            "events": asdict(s.events)}


def baseline_json(r: BaselineResult, pose_model: str) -> dict[str, Any]:
    """The baseline's results as data, for the F2 summary (no config: not a training run)."""
    return {"pose_model": pose_model, "thresholds": asdict(r.thresholds), **_scores_json(r)}


def _write_results(results_dir: Path, cfg: RunConfig, r: RunResult) -> None:
    (results_dir / f"{cfg.run_id}.json").write_text(
        json.dumps(results_json(cfg, r), indent=2) + "\n", encoding="utf-8")
    (results_dir / f"{cfg.run_id}.md").write_text(results_markdown(cfg, r), encoding="utf-8")


def baseline_markdown(r: BaselineResult, pose_model: str) -> str:
    return "\n".join([
        "# Threshold baseline", "",
        f"Pose model {pose_model}; thresholds fitted on the training windows: hip drop speed"
        f" ≥ {r.thresholds.speed} torso lengths/s, torso angle ≥ {r.thresholds.angle} rad.", "",
        *_measures(r)]) + "\n"


def run_queue(folder: Path, data_root: Path, results_dir: Path) -> int:
    """Run every config in `folder` in name order. A run whose results table exists is skipped,
    and an interrupted run resumes after its last finished epoch, so the queue can simply be
    started again after any stop. A failed run is logged and the queue moves on; returns 1 if
    any run failed."""
    failed = 0
    for path in sorted(folder.glob("*.toml")):
        try:
            cfg = load_config(path)
            target = results_dir / f"{cfg.run_id}.md"
            if target.exists():
                logger.info("%s: results exist, skipped", cfg.run_id)
                continue
            run = train(cfg, data_root, data_root / "runs" / cfg.run_id)
            _write_results(results_dir, cfg, run)  # the table last: it marks the run done
            logger.info("results written to %s", target)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            logger.error("run %s failed: %s", path.name, error)
            failed += 1
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train a fall model or fit the baseline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path)
    group.add_argument("--queue", type=Path, help="a folder of configs, run in name order")
    group.add_argument("--baseline", action="store_true")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pose-model", default="lite", help="for --baseline")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    data_root = args.data_root.expanduser()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.queue:
        return run_queue(args.queue, data_root, args.results_dir)
    try:
        if args.baseline:
            result = run_baseline(data_root, args.pose_model)
            text, name = baseline_markdown(result, args.pose_model), f"baseline-{args.pose_model}"
            (args.results_dir / f"{name}.json").write_text(
                json.dumps(baseline_json(result, args.pose_model), indent=2) + "\n",
                encoding="utf-8")
        else:
            cfg = load_config(args.config)
            run = train(cfg, data_root, data_root / "runs" / cfg.run_id)
            _write_results(args.results_dir, cfg, run)
            logger.info("results written to %s", args.results_dir / f"{cfg.run_id}.md")
            return 0
    except (OSError, ValueError, KeyError) as error:
        logger.error("run failed: %s", error)
        return 1
    (args.results_dir / f"{name}.md").write_text(text, encoding="utf-8")
    logger.info("results written to %s", args.results_dir / f"{name}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
