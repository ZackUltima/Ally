"""The one test-split run (F3b Task 6, D-68 decision 4).

Scores the shipped INT8 TCN (headline) and its float original on the test people, with and
without CAUCAFall's dim videos, as the fall class (fall or fallen) and as the fall label alone,
next to the threshold rule fitted in F2. Intervals resample whole people. The run refuses to
start when `results/f3b/test.json` exists: that file is the lock. This is the only module that
reads the test split; a test checks that no other source file mentions it.

Usage (after the PO says go):
  python -m ally_fall.final_test --config configs/f2-grid/tcn-omnifall16-b0.999-s0.toml \\
      --data-root ~/ally-data --int8 ~/ally-data/models/tcn-omnifall16-b0.999-s0-int8-ptq.tflite \\
      --confirm-open-test-split
  python -m ally_fall.final_test ... --split val --out-dir <folder>   # rehearsal, not locked
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from ally_fall.baseline import Thresholds, predict_from_features, rule_features
from ally_fall.build import window_features
from ally_fall.data import BuiltSplit, load_split, person_of, window_tensor, window_video
from ally_fall.labels import class_names, fall_classes, label_map
from ally_fall.metrics import Events, binary, count_events
from ally_fall.models import build_model
from ally_fall.omnifall import label_names
from ally_fall.ticks import TICK_S
from ally_fall.train import RunConfig, Scores, load_config, model_input, score

logger = logging.getLogger(__name__)

Predictor = Callable[[NDArray[np.float32]], NDArray[np.int64]]
FALL_CLASS, FALL_ONLY = "fall_class", "fall_only"
LOCK_NAME = "test.json"
CONDITIONS: dict[str, str] = {
    "all": "All {split} people",
    "no_dim": "Without CAUCAFall's dim videos",
}
_DIM_SENTENCE = re.compile(r"gives exactly person (\d+)'s (\d+) videos \(([^)]*)\)")


def parse_dim_videos(report: str) -> frozenset[str]:
    """The dim videos named in the F1 CAUCAFall report ("... person 8's 10 videos (A, B and C
    S8)"), as video names such as "FallForwardS8". The list was fixed by brightness on
    2026-10-20, before any model; it is read here, never computed from results."""
    found = _DIM_SENTENCE.search(report)
    if found is None:
        raise ValueError("no dim-video sentence found in the CAUCAFall report")
    person, count, listing = found.group(1), int(found.group(2)), found.group(3)
    items = [item.strip() for item in re.split(r",| and ", listing) if item.strip()]
    names = frozenset(re.sub(rf"\s*S{person}$", "", item) + f"S{person}" for item in items)
    if len(names) != count:
        raise ValueError(f"the report names {count} dim videos but {len(names)} were read: "
                         f"{sorted(names)}")
    return names


def dim_video_mask(video_ids: Sequence[str], datasets: Sequence[str],
                   names: frozenset[str]) -> NDArray[np.bool_]:
    """Per video: CAUCAFall and named in the dim list (the last part of the video id)."""
    return np.array([d == "caucafall" and v.rsplit("/", 1)[-1] in names
                     for v, d in zip(video_ids, datasets, strict=True)], dtype=np.bool_)


def thresholds_from(baseline_json: Path) -> Thresholds:
    """The threshold rule fitted on the training windows in F2 (`baseline-lite.json`)."""
    raw = json.loads(baseline_json.read_text(encoding="utf-8"))["thresholds"]
    return Thresholds(speed=float(raw["speed"]), angle=float(raw["angle"]))


@dataclass(frozen=True)
class Condition:
    people: int
    videos: int
    windows: int
    int8: dict[str, Scores]
    float_model: dict[str, Scores]
    rule: Scores


@dataclass(frozen=True)
class FinalTest:
    run_id: str = ""
    thresholds: Thresholds = Thresholds(0.0, 0.0)
    dim_videos_found: int = 0
    fall_agreement: float = 0.0
    label_agreement: float = 0.0
    conditions: dict[str, Condition] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> FinalTest:
        return cls()


def _sub(split: BuiltSplit, keep: NDArray[np.bool_]) -> BuiltSplit:
    """The split with only the windows in `keep`; the videos stay, so ids keep their meaning."""
    return replace(split, window_end=split.window_end[keep], y=split.y[keep])


def _both_scores(pred: NDArray[np.int64], split: BuiltSplit, fall: tuple[int, ...],
                 classes: tuple[str, ...]) -> dict[str, Scores]:
    """The fall class (fall or fallen), and the fall label alone."""
    return {
        FALL_CLASS: score(pred, np.isin(pred, fall), split, fall, classes),
        FALL_ONLY: score(pred, pred == fall[0], split, (fall[0], fall[0]), classes),
    }


def _predict_float(model: torch.nn.Module, split: BuiltSplit, cfg: RunConfig,
                   chunk: int = 2048) -> NDArray[np.int64]:
    model.eval()
    out: list[NDArray[np.int64]] = []
    with torch.no_grad():
        for start in range(0, len(split.window_end), chunk):
            idx = np.arange(start, min(start + chunk, len(split.window_end)))
            x = model_input(window_tensor(split, idx, cfg.window_ticks), cfg)
            out.append(model(torch.from_numpy(x)).argmax(1).numpy().astype(np.int64))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def _predict_rule(split: BuiltSplit, t: Thresholds, chunk: int = 4096) -> NDArray[np.bool_]:
    parts = []
    for start in range(0, len(split.window_end), chunk):
        idx = np.arange(start, min(start + chunk, len(split.window_end)))
        parts.append(predict_from_features(rule_features(window_tensor(split, idx)), t))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.bool_)


def _load(cfg: RunConfig, data_root: Path, split: str
          ) -> tuple[BuiltSplit, list[str], list[str], tuple[str, ...], tuple[int, ...]]:
    """The built split, its video ids and dataset names, the class names and the fall class."""
    names = label_names(data_root / "omnifall")
    path = data_root / "built" / cfg.pose_model / f"{split}.npz"
    data = load_split(path, label_map(names, cfg.label_set))
    with np.load(path, allow_pickle=False) as raw:
        video_ids, datasets = raw["video_id"].tolist(), raw["dataset"].tolist()
    return (data, video_ids, datasets, class_names(cfg.label_set, names),
            fall_classes(cfg.label_set, names))


def evaluate(cfg: RunConfig, data_root: Path, weights: Path, int8_predict: Predictor,
             thresholds: Thresholds, dim_names: frozenset[str], split: str = "test",
             lock: Path | None = None, expected_dim: int | None = None) -> FinalTest:
    """Score the INT8 model, the float model and the rule on `split` (all of them at once).

    `lock`, when given, is checked before anything is read. `expected_dim` is the number of dim
    videos the split must contain; a different count stops the run before any report is written.
    """
    if lock is not None and lock.exists():
        raise FileExistsError(f"{lock} exists: the test split was already opened once")
    data, video_ids, datasets, classes, fall = _load(cfg, data_root, split)
    dim_video = dim_video_mask(video_ids, datasets, dim_names)
    if expected_dim is not None and int(dim_video.sum()) != expected_dim:
        raise ValueError(f"found {int(dim_video.sum())} dim videos in the {split} split, "
                         f"expected {expected_dim}; nothing was written")
    dim_window = dim_video[window_video(data)]

    model = build_model(cfg.model, len(classes), cfg.hidden, None, cfg.window_ticks)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    float_pred = _predict_float(model, data, cfg)
    everything = np.arange(len(data.window_end), dtype=np.int64)
    int8_pred = int8_predict(model_input(window_tensor(data, everything, cfg.window_ticks), cfg))
    rule_pred = _predict_rule(data, thresholds)

    conditions: dict[str, Condition] = {}
    for key, keep in (("all", np.ones(len(data.window_end), dtype=np.bool_)),
                      ("no_dim", ~dim_window)):
        part = _sub(data, keep)
        conditions[key] = Condition(
            people=int(np.unique(person_of(part)).size),
            videos=int(np.unique(window_video(part)).size),
            windows=int(keep.sum()),
            int8=_both_scores(int8_pred[keep], part, fall, classes),
            float_model=_both_scores(float_pred[keep], part, fall, classes),
            rule=score(None, rule_pred[keep], part, fall, classes),
        )
    return FinalTest(
        run_id=cfg.run_id, thresholds=thresholds, dim_videos_found=int(dim_video.sum()),
        fall_agreement=float(np.mean(np.isin(float_pred, fall) == np.isin(int8_pred, fall))),
        label_agreement=float(np.mean(float_pred == int8_pred)), conditions=conditions)


# ------------------------------------------------------------------------------- reporting

def _scores_dict(s: Scores) -> dict[str, Any]:
    b = s.val
    return {"tp": b.tp, "fp": b.fp, "fn": b.fn, "tn": b.tn, "recall": b.recall,
            "precision": b.precision, "f1": b.f1, "specificity": b.specificity,
            "events": {"detected": s.events.detected, "missed": s.events.missed,
                       "false_alerts": s.events.false_alerts},
            "intervals": {k: list(v) for k, v in s.intervals.items()}}


def result_json(result: FinalTest, split: str) -> dict[str, Any]:
    return {
        "split": split, "run_id": result.run_id, "opened": datetime.date.today().isoformat(),
        "thresholds": {"speed": result.thresholds.speed, "angle": result.thresholds.angle},
        "dim_videos_found": result.dim_videos_found, "sources": result.sources,
        "float_int8_agreement": {"fall": result.fall_agreement, "label": result.label_agreement},
        "people": {k: c.people for k, c in result.conditions.items()},
        "conditions": {k: {
            "people": c.people, "videos": c.videos, "windows": c.windows,
            "int8": {w: _scores_dict(s) for w, s in c.int8.items()},
            "float": {w: _scores_dict(s) for w, s in c.float_model.items()},
            "rule": _scores_dict(c.rule)} for k, c in result.conditions.items()},
    }


def _row(name: str, s: Scores) -> str:
    b, ci, e = s.val, s.intervals, s.events
    span = lambda key: f"{ci[key][0]:.3f}–{ci[key][1]:.3f}"  # noqa: E731
    return (f"| {name} | {b.recall:.3f} ({span('recall')}) | {b.precision:.3f} "
            f"({span('precision')}) | {b.f1:.3f} ({span('f1')}) | {b.specificity:.3f} | "
            f"{e.detected} of {e.detected + e.missed} | {e.missed} | {e.false_alerts} | "
            f"{b.tp} / {b.fp} / {b.fn} / {b.tn} |")


_HEAD = ("| model | recall (95% by person) | precision | F1 | specificity | falls found | "
         "falls missed | false alerts | TP / FP / FN / TN |\n|---|---|---|---|---|---|---|---|---|")


def result_markdown(result: FinalTest, split: str) -> str:
    title = "F3b test-split result (opened once)" if split == "test" else \
        f"F3b rehearsal on the {split} split (not a test result)"
    lines = [
        f"# {title}", "",
        f"Run {result.run_id}; opened {datetime.date.today().isoformat()}. Headline: the INT8 "
        "file that ships (post-training quantisation, seed 0); the float original and the "
        "threshold rule are shown beside it. The fall class is fall or fallen; falls found "
        "counts whole fall events. Intervals are 95% percentile intervals from resampling "
        f"whole people. Rule: hip drop speed ≥ {result.thresholds.speed} torso lengths/s and "
        f"torso angle ≥ {result.thresholds.angle} rad, fitted on the training windows in F2.",
        "", f"Float and INT8 give the same fall decision on {result.fall_agreement:.4f} of "
        f"these windows and the same label on {result.label_agreement:.4f}.", ""]
    for key, label in CONDITIONS.items():
        c = result.conditions[key]
        heading = f"{label.format(split=split)}: {c.people} people, {c.videos} videos"
        lines += [f"## {heading}, {c.windows} windows", "",
                  "### Fall class (fall or fallen)", "", _HEAD,
                  _row("INT8 TCN (headline)", c.int8[FALL_CLASS]),
                  _row("float TCN", c.float_model[FALL_CLASS]),
                  _row("threshold rule", c.rule), "",
                  "### Fall label alone", "", _HEAD,
                  _row("INT8 TCN (headline)", c.int8[FALL_ONLY]),
                  _row("float TCN", c.float_model[FALL_ONLY]), ""]
    lines += ["The \"without\" rows drop all of CAUCAFall person 8's videos (the dim list of "
              f"2026-10-20, {result.dim_videos_found} videos found), so that condition has one "
              "person fewer.", ""]
    return "\n".join(lines)


def write_reports(result: FinalTest, out_dir: Path, split: str) -> None:
    """Writes the report and, for the test split, the lock `test.json` (created first, and only
    if it does not exist). A rehearsal on another split writes `rehearsal-<split>.*` and can be
    repeated."""
    stem = "test" if split == "test" else f"rehearsal-{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = "x" if split == "test" else "w"
    try:
        with (out_dir / f"{stem}.json").open(mode, encoding="utf-8") as lock:
            lock.write(json.dumps(result_json(result, split), indent=2) + "\n")
    except FileExistsError as error:
        raise FileExistsError(f"{out_dir / f'{stem}.json'} exists: the test split was already "
                              "opened once; deleting it is the PO's decision") from error
    (out_dir / f"{stem}.md").write_text(result_markdown(result, split), encoding="utf-8")


# ------------------------------------------------------------- error analysis (after the score)

def precision_at(recall: float, specificity: float, prevalence: float) -> float:
    """The precision a model with this recall and specificity would have if `prevalence` of the
    windows were fall class."""
    tp, fp = recall * prevalence, (1.0 - specificity) * (1.0 - prevalence)
    return tp / (tp + fp) if tp + fp > 0 else 0.0


@dataclass(frozen=True)
class Slice:
    """The INT8 model's fall-class results on the windows of one dataset or one person."""

    windows: int
    fall_windows: int
    tp: int
    fp: int
    fn: int
    tn: int
    found: int
    missed: int
    false_alerts: int
    no_pose: float  # mean share of ticks with no pose in these windows


@dataclass(frozen=True)
class LabelRow:
    label: str
    windows: int
    flagged: int  # windows of this (non-fall) label that the model called a fall


@dataclass(frozen=True)
class PoseBin:
    name: str
    windows: int
    fall_windows: int
    recall: float
    non_fall_windows: int
    flagged_rate: float  # share of non-fall windows called a fall
    false_positives: int
    share_of_false_positives: float


@dataclass(frozen=True)
class Breakdown:
    run_id: str
    totals: Slice
    by_dataset: dict[str, Slice]
    by_person: dict[str, Slice]
    false_positive_labels: list[LabelRow]
    pose_bins: list[PoseBin]
    prevalence_test: float
    prevalence_val: float
    precision_at_val_prevalence: float
    kind: str = "error_analysis"


POSE_BINS: tuple[tuple[str, float], ...] = (
    ("no ticks lost", 0.0), ("up to a fifth of the ticks", 0.2),
    ("a fifth to half of the ticks", 0.5), ("over half of the ticks", 1.0))


def _missing_share(data: BuiltSplit, ticks: int, chunk: int = 4096) -> NDArray[np.float64]:
    """Per window: the share of its ticks in which no pose was found (all visibilities zero)."""
    parts = []
    for start in range(0, len(data.window_end), chunk):
        end = data.window_end[start:start + chunk]
        windows = window_features(data.features, end, ticks)  # (n, ticks, 33, 3)
        parts.append((windows[..., 2].max(axis=-1) == 0).mean(axis=1))
    return np.concatenate(parts) if parts else np.zeros(0)


def _event_counts(pred_fall: NDArray[np.bool_], split: BuiltSplit,
                  fall: tuple[int, ...]) -> Events:
    detected = missed = false_alerts = 0
    video = window_video(split)
    for v in np.unique(video):
        idx = np.flatnonzero(video == v)
        times = (split.window_end[idx] - split.video_start[v]) * TICK_S
        e = count_events(times, pred_fall[idx], split.y[idx], fall[0], fall[1])
        detected, missed, false_alerts = (detected + e.detected, missed + e.missed,
                                          false_alerts + e.false_alerts)
    return Events(detected, missed, false_alerts)


def _slice(data: BuiltSplit, pred_fall: NDArray[np.bool_], missing: NDArray[np.float64],
           fall: tuple[int, ...], keep: NDArray[np.bool_]) -> Slice:
    part = _sub(data, keep)
    true_fall = np.isin(part.y, fall)
    b = binary(pred_fall[keep], true_fall)
    e = _event_counts(pred_fall[keep], part, fall)
    return Slice(windows=int(keep.sum()), fall_windows=int(true_fall.sum()), tp=b.tp, fp=b.fp,
                 fn=b.fn, tn=b.tn, found=e.detected, missed=e.missed, false_alerts=e.false_alerts,
                 no_pose=float(missing[keep].mean()) if keep.any() else 0.0)


def breakdown(cfg: RunConfig, data_root: Path, int8_predict: Predictor,
              reference: dict[str, Any], split: str = "test") -> Breakdown:
    """Where the INT8 model's errors on `split` are, by dataset, person, true label and pose
    loss. It first reproduces the saved result (`reference` is the parsed `test.json`) and stops
    if the totals differ, so it is the same run. An error analysis, not a new score."""
    data, _, datasets, classes, fall = _load(cfg, data_root, split)
    everything = np.arange(len(data.window_end), dtype=np.int64)
    pred = int8_predict(model_input(window_tensor(data, everything, cfg.window_ticks), cfg))
    pred_fall, true_fall = np.isin(pred, fall), np.isin(data.y, fall)
    missing = _missing_share(data, cfg.window_ticks)
    totals = _slice(data, pred_fall, missing, fall, np.ones(len(pred), dtype=np.bool_))
    saved = reference["conditions"]["all"]["int8"][FALL_CLASS]
    if any(getattr(totals, k) != saved[k] for k in ("tp", "fp", "fn", "tn")):
        raise ValueError("the analysis does not reproduce the saved test result "
                         f"(TP/FP/FN/TN {totals.tp}/{totals.fp}/{totals.fn}/{totals.tn} against "
                         f"{saved['tp']}/{saved['fp']}/{saved['fn']}/{saved['tn']})")
    video = window_video(data)
    window_dataset = np.array(datasets)[video]
    window_person = np.array([f"{datasets[v]}/{data.subject[v]}" for v in video])
    by_dataset = {d: _slice(data, pred_fall, missing, fall, window_dataset == d)
                  for d in sorted(set(datasets))}
    by_person = {p: _slice(data, pred_fall, missing, fall, window_person == p)
                 for p in sorted(set(window_person.tolist()))}

    rows = []
    for index, label in enumerate(classes):
        here = data.y == index
        if index not in fall and here.any():
            rows.append(LabelRow(label, int(here.sum()), int((pred_fall & here).sum())))
    rows.sort(key=lambda r: (-r.flagged, r.label))

    lower, bins = 0.0, []
    for name, upper in POSE_BINS:
        in_bin = (missing == 0) if upper == 0.0 else (missing > lower) & (missing <= upper)
        lower = upper
        fell, other = in_bin & true_fall, in_bin & ~true_fall
        flagged = int((pred_fall & other).sum())
        bins.append(PoseBin(
            name=name, windows=int(in_bin.sum()), fall_windows=int(fell.sum()),
            recall=float((pred_fall & fell).sum() / fell.sum()) if fell.any() else 0.0,
            non_fall_windows=int(other.sum()),
            flagged_rate=flagged / int(other.sum()) if other.any() else 0.0,
            false_positives=flagged,
            share_of_false_positives=flagged / totals.fp if totals.fp else 0.0))

    names = label_names(data_root / "omnifall")
    val = load_split(data_root / "built" / cfg.pose_model / "val.npz",
                     label_map(names, cfg.label_set))
    recall = totals.tp / (totals.tp + totals.fn) if totals.tp + totals.fn else 0.0
    specificity = totals.tn / (totals.tn + totals.fp) if totals.tn + totals.fp else 0.0
    p_val = float(np.isin(val.y, fall).mean())
    return Breakdown(
        run_id=cfg.run_id, totals=totals, by_dataset=by_dataset, by_person=by_person,
        false_positive_labels=rows, pose_bins=bins, prevalence_test=float(true_fall.mean()),
        prevalence_val=p_val, precision_at_val_prevalence=precision_at(recall, specificity, p_val))


def _slice_row(name: str, s: Slice) -> str:
    precision = s.tp / (s.tp + s.fp) if s.tp + s.fp else 0.0
    recall = s.tp / (s.tp + s.fn) if s.tp + s.fn else 0.0
    return (f"| {name} | {s.windows} | {s.fall_windows} | {s.tp} / {s.fp} / {s.fn} / {s.tn} | "
            f"{precision:.3f} | {recall:.3f} | {s.found} of {s.found + s.missed} | "
            f"{s.false_alerts} | {100 * s.no_pose:.1f}% |")


_SLICE_HEAD = ("| {what} | windows | fall-class windows | TP / FP / FN / TN | precision | recall "
               "| falls found | false alerts | ticks with no pose |\n"
               "|---|---|---|---|---|---|---|---|---|")


def breakdown_markdown(b: Breakdown) -> str:
    test_precision = b.totals.tp / max(b.totals.tp + b.totals.fp, 1)
    lines = [
        "# F3b test-split error analysis (INT8 TCN, fall class)", "",
        "**This is an error analysis made after the test result was seen. It is not a new "
        "score:** the reported test result stays in `test.md`. It reproduces that result "
        "exactly before it breaks it down, and nothing is tuned on it. Fixes are judged on "
        "validation and on new data, never on these 9 people.", "",
        f"Run {b.run_id}. Fall-class windows are {100 * b.prevalence_test:.1f}% of the test "
        f"windows and {100 * b.prevalence_val:.1f}% of the validation windows. With the test "
        "recall and specificity but the validation share of falls, precision would be "
        f"{b.precision_at_val_prevalence:.3f} (test: {test_precision:.3f}); the rest of the "
        "drop is not explained by how many falls there are.", "",
        "## By dataset", "", _SLICE_HEAD.format(what="dataset")]
    lines += [_slice_row(k, s) for k, s in b.by_dataset.items()]
    lines += ["", "## By person", "", _SLICE_HEAD.format(what="dataset / person")]
    lines += [_slice_row(k, s) for k, s in b.by_person.items()]
    lines += ["", "## What the false alarms really were (true label of windows called a fall)", "",
              "| true label | windows | called a fall | share |", "|---|---|---|---|"]
    lines += [f"| {r.label} | {r.windows} | {r.flagged} | {100 * r.flagged / r.windows:.1f}% |"
              for r in b.false_positive_labels]
    lines += ["", "## By how much of the window had no pose", "",
              "| ticks with no pose | windows | recall on fall windows | non-fall windows | "
              "called a fall | false positives | share of all false positives |",
              "|---|---|---|---|---|---|---|"]
    lines += [f"| {p.name} | {p.windows} | {p.recall:.3f} | {p.non_fall_windows} | "
              f"{100 * p.flagged_rate:.1f}% | {p.false_positives} | "
              f"{100 * p.share_of_false_positives:.1f}% |" for p in b.pose_bins]
    return "\n".join(lines) + "\n"


def write_breakdown(b: Breakdown, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test-breakdown.json").write_text(
        json.dumps(asdict(b), indent=2) + "\n", encoding="utf-8")
    (out_dir / "test-breakdown.md").write_text(breakdown_markdown(b), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - needs the real data
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--int8", type=Path, required=True, help="the shipped .tflite file")
    parser.add_argument("--baseline", type=Path, default=Path("results/runs/baseline-lite.json"))
    parser.add_argument("--dim-report", type=Path, default=Path("results/data/caucafall.md"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/f3b"))
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--breakdown", action="store_true",
                        help="error analysis of the saved test result (needs test.json)")
    parser.add_argument("--confirm-open-test-split", action="store_true",
                        help="required for --split test: it can be opened only once")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.breakdown:
        from ally_fall.export import run_tflite  # lazy: needs the LiteRT runtime

        reference = json.loads((args.out_dir / LOCK_NAME).read_text(encoding="utf-8"))
        cfg = load_config(args.config)
        write_breakdown(breakdown(
            cfg, args.data_root.expanduser(),
            lambda windows: run_tflite(args.int8, windows).preds, reference), args.out_dir)
        logger.info("error analysis written to %s", args.out_dir)
        return 0
    if args.split == "test" and not args.confirm_open_test_split:
        logger.error("the test split opens once; add --confirm-open-test-split")
        return 2
    from ally_fall.export import run_tflite  # lazy: needs the LiteRT runtime

    cfg = load_config(args.config)
    data_root = args.data_root.expanduser()
    dim = parse_dim_videos(args.dim_report.read_text(encoding="utf-8"))
    lock = args.out_dir / LOCK_NAME if args.split == "test" else None
    result = evaluate(
        cfg, data_root, data_root / "runs" / cfg.run_id / "best.pt",
        lambda windows: run_tflite(args.int8, windows).preds, thresholds_from(args.baseline), dim,
        split=args.split, lock=lock, expected_dim=len(dim) if args.split == "test" else None)
    result = replace(result, sources={
        "float_weights": str(data_root / "runs" / cfg.run_id / "best.pt"),
        "int8_file": args.int8.name, "baseline": str(args.baseline),
        "dim_report": str(args.dim_report)})
    write_reports(result, args.out_dir, args.split)
    logger.info("written to %s", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
