"""The F3b model-choice rule, applied to saved validation results (F3b Task 1, D-68 decision 1).

The pretrained ST-GCN and the TCN (omnifall16, lite pose, kept beta, seeds 0-2) are compared on the
validation people only, in this fixed order:
  (i)   falls found of 57 (mean over seeds): if the two differ by 2 falls or more, the higher wins;
  (ii)  otherwise the higher window F1 wins (a tie takes the smaller model);
  (iv)  if the pretrained model won but is not at least 0.02 F1 above the TCN, the TCN is taken;
  (iii) the winner must pass the INT8 gate (Task 5), else the other model is taken. Until the
        gate has run the choice is provisional.
No model code runs here, and a path under the test split is refused.

Usage: python -m ally_fall.choose [--results-dir results/runs] [--out-dir results/f3b]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from ally_fall.models import build_model
from ally_fall.summary import PRETRAINED, SEEDS
from ally_fall.train import RESULTS_DIR

TCN = "tcn"
LABEL_SET = "omnifall16"
POSE = "lite"
BETA = 0.999
FALLS_GAP = 2.0
F1_GAP = 0.02
EPS = 1e-9
REPORT_NAME = "model-choice.md"
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "f3b"


@dataclass(frozen=True)
class Candidate:
    """One model's validation results, one value per seed."""

    name: str
    f1: dict[int, float]
    recall: dict[int, float]
    precision: dict[int, float]
    detected: dict[int, int]
    missed: dict[int, int]
    false_alerts: dict[int, int]
    parameters: int

    @property
    def mean_f1(self) -> float:
        return _mean(self.f1)

    @property
    def f1_range(self) -> float:
        return max(self.f1.values()) - min(self.f1.values())

    @property
    def mean_recall(self) -> float:
        return _mean(self.recall)

    @property
    def mean_precision(self) -> float:
        return _mean(self.precision)

    @property
    def mean_detected(self) -> float:
        return _mean(self.detected)

    @property
    def mean_missed(self) -> float:
        return _mean(self.missed)

    @property
    def mean_false_alerts(self) -> float:
        return _mean(self.false_alerts)

    @property
    def total_falls(self) -> int:
        totals = {self.detected[s] + self.missed[s] for s in self.detected}
        if len(totals) != 1:
            raise ValueError(f"{self.name}: seeds disagree on the falls: {sorted(totals)}")
        return totals.pop()


@dataclass(frozen=True)
class Choice:
    """The rule's answer: the model, the branch that decided it and one plain sentence."""

    winner: str
    branch: str
    why: str
    provisional: bool


def _mean(values: Mapping[int, float]) -> float:
    return sum(values.values()) / len(values)


def choose(
    pretrained: Candidate, tcn: Candidate, int8_ok: Mapping[str, bool] | None = None
) -> Choice:
    """Applies the rule in its stated order; `int8_ok` maps a model name to its INT8 gate result."""
    gap = pretrained.mean_detected - tcn.mean_detected
    f1_gap = pretrained.mean_f1 - tcn.mean_f1
    if abs(gap) >= FALLS_GAP - EPS:
        first, branch = (pretrained, "falls_found") if gap > 0 else (tcn, "falls_found")
        why = (f"{first.name} finds {first.mean_detected:.1f} of {first.total_falls} falls, "
               f"{abs(gap):.1f} more than the other model (rule: 2 or more decides)")
    else:
        first, branch = (pretrained, "window_f1") if f1_gap > EPS else (tcn, "window_f1")
        why = (f"the two find {abs(gap):.1f} falls apart (under 2), so window F1 decides: "
               f"{first.name} {first.mean_f1:.3f}")
    if first is pretrained and f1_gap < F1_GAP - EPS:
        first, branch = tcn, "not_clearly_better"
        why = (f"the pretrained model is {f1_gap:+.3f} F1 from the TCN, under the 0.02 needed, "
               f"so the {_size_ratio(pretrained, tcn):.0f} times smaller TCN is taken")
    return _gate(first, pretrained if first is tcn else tcn, branch, why, int8_ok)


def _size_ratio(big: Candidate, small: Candidate) -> float:
    return big.parameters / small.parameters


def _gate(
    first: Candidate, other: Candidate, branch: str, why: str, int8_ok: Mapping[str, bool] | None
) -> Choice:
    result = (int8_ok or {}).get(first.name)
    if result is None:
        return Choice(first.name, branch, why, provisional=True)
    if result:
        return Choice(first.name, branch, why, provisional=False)
    if (int8_ok or {}).get(other.name):
        return Choice(other.name, "int8_gate",
                      f"{first.name} failed the INT8 gate, so {other.name} is taken ({why})", False)
    return Choice(first.name, "int8_gate_none_passed",
                  f"neither model passed the INT8 gate; the PO decides next ({why})", False)


def refuse_test_split(path: Path) -> None:
    """Raises for a folder or file that belongs to the test split (named `test`)."""
    if "test" in {part.lower() for part in path.parts} or path.stem.lower() == "test":
        raise ValueError(f"{path}: only validation results may be read here, not the test split")


def read_validation(path: Path) -> dict[str, Any]:
    """One run's JSON; anything under the test split is refused (the test split is opened once)."""
    refuse_test_split(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and str(data.get("split", "val")).lower() == "test":
        raise ValueError(f"{path}: only validation results may be read here, not the test split")
    return data  # type: ignore[no-any-return]


@cache
def _parameters(model: str, n_classes: int, hidden: int) -> int:
    net = build_model(model, n_classes, hidden)  # type: ignore[arg-type]
    return sum(p.numel() for p in net.parameters())


def load_candidates(results_dir: Path) -> tuple[Candidate, Candidate]:
    """The pretrained ST-GCN and the TCN, picked by each run's config and never by its ID."""
    refuse_test_split(results_dir)
    runs = [read_validation(p) for p in sorted(results_dir.glob("*.json"))]
    runs = [r for r in runs if isinstance(r, dict) and "config" in r]
    return _candidate(runs, PRETRAINED, pretrained=True), _candidate(runs, TCN, pretrained=False)


def _candidate(runs: list[dict[str, Any]], model: str, *, pretrained: bool) -> Candidate:
    chosen = [r for r in runs if _matches(r["config"], model, pretrained)]
    seeds = sorted(r["config"]["seed"] for r in chosen)
    if seeds != list(SEEDS):
        raise ValueError(f"{model}: needs one run per seed {list(SEEDS)}, found {seeds}")
    first = chosen[0]
    return Candidate(
        name=model,
        f1={r["config"]["seed"]: float(r["val"]["f1"]) for r in chosen},
        recall={r["config"]["seed"]: float(r["val"]["recall"]) for r in chosen},
        precision={r["config"]["seed"]: float(r["val"]["precision"]) for r in chosen},
        detected={r["config"]["seed"]: int(r["events"]["detected"]) for r in chosen},
        missed={r["config"]["seed"]: int(r["events"]["missed"]) for r in chosen},
        false_alerts={r["config"]["seed"]: int(r["events"]["false_alerts"]) for r in chosen},
        parameters=_parameters(
            model, len(first["train_counts"]), int(first["config"].get("hidden", 64))
        ),
    )


def _matches(config: dict[str, Any], model: str, pretrained: bool) -> bool:
    return (config["model"] == model and config["label_set"] == LABEL_SET
            and config.get("pose_model", POSE) == POSE and float(config["beta"]) == BETA
            and bool(config.get("checkpoint", "")) == pretrained)


def choice_markdown(pretrained: Candidate, tcn: Candidate, pick: Choice) -> str:
    """The report: the two models side by side, the branch taken and one sentence on why."""
    total = pretrained.total_falls

    def row(label: str, a: str, b: str) -> str:
        return f"| {label} | {a} | {b} |"

    def seeds(values: Mapping[int, float]) -> str:
        return ", ".join(f"{values[s]:g}" for s in sorted(values))

    status = "Provisional" if pick.provisional else "Final"
    gate = ("not yet run (Task 5); the choice above is provisional until it has" if pick.provisional
            else "run; see `int8.md`")
    lines = [
        f"# F3b model choice ({status.lower()}, validation only)",
        "",
        f"**{status} choice: {pick.winner}**, by branch `{pick.branch}`: {pick.why}.",
        "",
        "The rule was fixed before any result (D-68 decision 1): falls found first (a gap of 2 "
        "or more decides), then window F1, then the smaller model unless the pretrained one is "
        "at least 0.02 F1 above it, then the INT8 gate. Validation people only; the test split "
        "is untouched. Falls are the fall class (fall or fallen); "
        f"{total} falls in the validation videos; means over seeds 0-2.",
        "",
        f"| | pretrained ST-GCN (`{pretrained.name}`) | TCN (`{tcn.name}`) |",
        "|---|---|---|",
        row("window F1, mean (range over seeds)",
            f"{pretrained.mean_f1:.3f} ({pretrained.f1_range:.3f})",
            f"{tcn.mean_f1:.3f} ({tcn.f1_range:.3f})"),
        row("recall, mean", f"{pretrained.mean_recall:.3f}", f"{tcn.mean_recall:.3f}"),
        row("precision, mean", f"{pretrained.mean_precision:.3f}", f"{tcn.mean_precision:.3f}"),
        row(f"falls found of {total}, mean",
            f"{pretrained.mean_detected:.1f}", f"{tcn.mean_detected:.1f}"),
        row("falls found per seed", seeds(pretrained.detected), seeds(tcn.detected)),
        row("falls missed, mean", f"{pretrained.mean_missed:.1f}", f"{tcn.mean_missed:.1f}"),
        row("false alerts, mean",
            f"{pretrained.mean_false_alerts:.1f}", f"{tcn.mean_false_alerts:.1f}"),
        row("parameters", f"{pretrained.parameters:,}", f"{tcn.parameters:,}"),
        "",
        f"INT8 gate: {gate}.",
        "",
    ]
    return "\n".join(lines)


def write_choice(
    results_dir: Path, out_dir: Path, int8_ok: Mapping[str, bool] | None = None
) -> Path:
    pretrained, tcn = load_candidates(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / REPORT_NAME
    pick = choose(pretrained, tcn, int8_ok)
    path.write_text(choice_markdown(pretrained, tcn, pick), encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    path = write_choice(args.results_dir, args.out_dir)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
