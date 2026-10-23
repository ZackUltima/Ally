"""The F2 sprint summary and the selection rules fixed before any result (F2 Task 7 Step 6, D-62).

Everything is validation fall-class F1 (fall or fallen), mean over seeds 0-2; "range" is max - min
over the 3 seeds, and a configuration with fewer than 3 finished seeds gets no verdict.
- Decision 6: within one label set, the pretrained ST-GCN beats only if its mean is higher than
  the best from-scratch model's (TCN, GRU, light ST-GCN) by more than that model's range.
- Rule 1 (decision 7): the control trains on the label set where the pretrained model's mean is
  higher; a tie takes groups5.
- Rule 2 (decision 8): the best configuration is the highest mean among the from-scratch models
  and the pretrained model where it beats; a tie takes fewer parameters; never the control.
- Rule 3 (decision 8): full poses are kept only if they beat lite by more than lite's range.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from ally_fall.labels import LabelSet
from ally_fall.models import build_model
from ally_fall.splits import CROSS_SUBJECT

SEEDS: tuple[int, ...] = (0, 1, 2)
SCRATCH: tuple[str, ...] = ("tcn", "gru", "stgcn")
PRETRAINED = "stgcn_pt"
LABEL_SETS: tuple[LabelSet, ...] = ("omnifall16", "groups5")
SUMMARY_NAME = "f2-summary.md"
NOISE = 1e-9  # float slack, so a gap equal to the range counts as equal, not as more
DATASET_NAMES = {"up_fall": "UP-Fall", "caucafall": "CAUCAFall", "le2i": "Le2i",
                 "GMDCSA24": "GMDCSA-24"}


@dataclass(frozen=True)
class Key:
    """One configuration; its seeds are grouped under it."""

    model: str
    label_set: str
    pose_model: str
    beta: float
    pretrained: bool


@dataclass(frozen=True)
class Group:
    key: Key
    f1: dict[int, float]
    recall: dict[int, float]
    precision: dict[int, float]
    n_classes: int
    hidden: int

    @property
    def complete(self) -> bool:
        return set(self.f1) >= set(SEEDS)

    @property
    def mean(self) -> float:
        return sum(self.f1.values()) / len(self.f1)

    @property
    def spread(self) -> float:
        return max(self.f1.values()) - min(self.f1.values())

    @property
    def is_control(self) -> bool:
        return self.key.model == PRETRAINED and not self.key.pretrained


@dataclass(frozen=True)
class Verdict:
    label_set: str
    pretrained: Group
    scratch: Group

    @property
    def beats(self) -> bool:
        return self.pretrained.mean - self.scratch.mean > self.scratch.spread + NOISE


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    """Every run's JSON (files with a config); the baseline and other files are skipped."""
    out = []
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "config" in data:
            out.append(data)
    return out


def group_results(results: list[dict[str, Any]]) -> list[Group]:
    """Runs grouped by configuration, read from each run's config (never from its ID)."""
    found: dict[Key, list[dict[str, Any]]] = {}
    for r in results:
        c = r["config"]
        key = Key(c["model"], c["label_set"], c.get("pose_model", "lite"), float(c["beta"]),
                  bool(c.get("checkpoint", "")))
        found.setdefault(key, []).append(r)
    return [Group(key, {r["config"]["seed"]: float(r["val"]["f1"]) for r in rs},
                  {r["config"]["seed"]: float(r["val"]["recall"]) for r in rs},
                  {r["config"]["seed"]: float(r["val"]["precision"]) for r in rs},
                  len(rs[0]["train_counts"]), int(rs[0]["config"].get("hidden", 64)))
            for key, rs in sorted(found.items(), key=lambda kv: _order(kv[0]))]


def _order(k: Key) -> tuple[float, int, int, bool, bool]:
    """Table order: β, label set, model, lite before full, pretrained before control."""
    models = (*SCRATCH, PRETRAINED)
    rank = models.index(k.model) if k.model in models else len(models)
    sets = LABEL_SETS.index(k.label_set) if k.label_set in LABEL_SETS else len(LABEL_SETS)
    return (k.beta, sets, rank, k.pose_model != "lite", not k.pretrained)


@cache
def _params(model: str, n_classes: int, hidden: int) -> int:
    net = build_model(model, n_classes, hidden)  # type: ignore[arg-type]
    return sum(p.numel() for p in net.parameters())


def param_count(g: Group) -> int:
    return _params(g.key.model, g.n_classes, g.hidden)


def _get(groups: list[Group], key: Key) -> Group | None:
    g = next((g for g in groups if g.key == key), None)
    return g if g is not None and g.complete else None


def control_label_set(groups: list[Group], beta: float) -> LabelSet | None:
    """Rule 1; None until both pretrained configurations have all seeds."""
    pt = {s: _get(groups, Key(PRETRAINED, s, "lite", beta, True)) for s in LABEL_SETS}
    if pt["omnifall16"] is None or pt["groups5"] is None:
        return None
    return "omnifall16" if pt["omnifall16"].mean > pt["groups5"].mean + NOISE else "groups5"


def decision_6(groups: list[Group], beta: float, label_set: str) -> Verdict | None:
    """Decision 6 within one label set; None until every model there has all seeds."""
    pt = _get(groups, Key(PRETRAINED, label_set, "lite", beta, True))
    scratch = [_get(groups, Key(m, label_set, "lite", beta, False)) for m in SCRATCH]
    if pt is None or any(g is None for g in scratch):
        return None
    best = max((g for g in scratch if g is not None), key=lambda g: (g.mean, -param_count(g)))
    return Verdict(label_set, pt, best)


def best_configuration(groups: list[Group], beta: float) -> Group | None:
    """Rule 2; None until decision 6 can be applied in both label sets."""
    verdicts = [decision_6(groups, beta, s) for s in LABEL_SETS]
    if any(v is None for v in verdicts):
        return None
    candidates = [g for g in groups if g.key.beta == beta and g.key.pose_model == "lite"
                  and g.complete and g.key.model in SCRATCH]
    candidates += [v.pretrained for v in verdicts if v is not None and v.beats]
    return max(candidates, key=lambda g: (g.mean, -param_count(g)))


def pose_choice(groups: list[Group], lite: Group) -> str | None:
    """Rule 3; None until the full-pose runs of the same configuration have all seeds."""
    k = lite.key
    full = _get(groups, Key(k.model, k.label_set, "full", k.beta, k.pretrained))
    if full is None:
        return None
    return "full" if full.mean - lite.mean > lite.spread + NOISE else "lite"


def validation_people() -> list[str]:
    return [f"{DATASET_NAMES.get(d, d)} subject {n}"
            for d, split in CROSS_SUBJECT.items() for n in sorted(split["val"])]


def _baseline_row(results_dir: Path, pose: str) -> str:
    path = results_dir / f"baseline-{pose}.json"
    if not path.exists():
        return f"| {pose} | pending | | | |"
    data = json.loads(path.read_text(encoding="utf-8"))
    v, ci = data["val"], data["intervals"]["f1"]
    return (f"| {pose} | {v['recall']:.3f} | {v['precision']:.3f} | {v['f1']:.3f}"
            f" ({ci[0]:.3f}–{ci[1]:.3f}) | {v['specificity']:.3f} |")


def _start(g: Group) -> str:
    if g.key.model != PRETRAINED:
        return "random"
    return "pretrained" if g.key.pretrained else "random (control)"


def _run_row(g: Group) -> str:
    def mean(d: dict[int, float]) -> float:
        return sum(d.values()) / len(d)

    return (f"| {g.key.model} | {_start(g)} | {g.key.label_set} | {g.key.pose_model}"
            f" | {g.key.beta:g} | {len(g.f1)} of {len(SEEDS)} | {g.mean:.3f} | {g.spread:.3f}"
            f" | {mean(g.recall):.3f} | {mean(g.precision):.3f} | {param_count(g):,} |")


def _name(g: Group) -> str:
    return f"{g.key.model}, {_start(g)}, {g.key.label_set}"


def _decision_lines(groups: list[Group], beta: float | None) -> list[str]:
    if beta is None:
        return ["- **Decision 6:** pending (the β runs are not finished)",
                "- **Control run label set (rule 1):** pending",
                "- **Pretrained vs random start (decision 7):** pending",
                "- **Best configuration (rule 2):** pending",
                "- **Pose model (rule 3):** pending"]
    lines = []
    for s in LABEL_SETS:
        v = decision_6(groups, beta, s)
        if v is None:
            lines.append(f"- **Decision 6,** {s}: pending (not every model has 3 seeds)")
            continue
        gap = v.pretrained.mean - v.scratch.mean
        word = "pretrained beats" if v.beats else "tie, from-scratch kept"
        lines.append(f"- **Decision 6,** {s}: {word} (pretrained {v.pretrained.mean:.3f} vs"
                     f" {v.scratch.key.model} {v.scratch.mean:.3f}: gap {gap:+.3f}, needs more"
                     f" than its range {v.scratch.spread:.3f})")
    ls = control_label_set(groups, beta)
    lines.append(f"- **Control run label set (rule 1):** {ls or 'pending'}")
    control = _get(groups, Key(PRETRAINED, ls, "lite", beta, False)) if ls else None
    pt = _get(groups, Key(PRETRAINED, ls, "lite", beta, True)) if ls else None
    if control is None or pt is None:
        lines.append("- **Pretrained vs random start (decision 7):** pending")
    else:
        lines.append(f"- **Pretrained vs random start (decision 7),** {ls}: pretrained"
                     f" {pt.mean:.3f} vs random start {control.mean:.3f}"
                     f" (gap {pt.mean - control.mean:+.3f}; ranges {pt.spread:.3f} and"
                     f" {control.spread:.3f})")
    best = best_configuration(groups, beta)
    lines.append("- **Best configuration (rule 2):** "
                 + (f"{_name(best)} (mean {best.mean:.3f}, {param_count(best):,} parameters)"
                    if best else "pending"))
    pose = pose_choice(groups, best) if best else None
    lines.append(f"- **Pose model (rule 3):** {pose or 'pending'}")
    return lines


def summary_markdown(results_dir: Path, beta: float | None) -> str:
    groups = group_results(load_results(results_dir))
    return "\n".join([
        "# Sprint F2 summary: fall model training", "",
        "Validation only: the test split is opened once, in F3. Fall class = fall ∪ fallen;"
        " F1, recall and precision per window, mean over seeds 0–2; range = max − min over the"
        " seeds. Selection rules fixed before any result (D-62, PO 2026-10-23).", "",
        "Validation people: " + ", ".join(validation_people()) + ".", "",
        f"Kept β: {f'{beta:g} (`beta-choice.md`)' if beta is not None else 'pending'}", "",
        "## Threshold baseline", "",
        "| pose | recall | precision | F1 (95% interval by person) | specificity |",
        "|---|---|---|---|---|", _baseline_row(results_dir, "lite"),
        _baseline_row(results_dir, "full"), "",
        "## Runs", "",
        "| model | start | label set | pose | β | seeds | F1 mean | F1 range | recall mean"
        " | precision mean | parameters |", "|---|---|---|---|---|---|---|---|---|---|---|",
        *([_run_row(g) for g in groups] or ["| pending | | | | | | | | | | |"]), "",
        "## Decisions", "", *_decision_lines(groups, beta), ""])


def write_summary(results_dir: Path, beta: float | None) -> Path:
    path = results_dir / SUMMARY_NAME
    results_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(summary_markdown(results_dir, beta), encoding="utf-8")
    return path
