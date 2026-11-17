"""Posture of the fall model on the validation windows (P1 Task 3, D-72).

The shipped TCN (seeds 0-2) is scored on the validation split only: its most likely posture
against the posture of the true OmniFall label, with F1 per posture, by dataset, by person and
by how many ticks of the window had no pose. The bar (a proposal, D-72): upright, sitting and
lying reach F1 >= 0.80 on every seed. The split that is opened once at the end is never read
here, and a path under a folder with that name is refused.

Usage (WSL, CPU, minutes):
  python -m ally_fall.posture_report --data-root ~/ally-data --config-dir configs/f2-grid \\
      --out results/posture/validation.md
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ally_fall.build import window_features
from ally_fall.choose import refuse_test_split
from ally_fall.data import BuiltSplit, window_tensor, window_video
from ally_fall.live.scorer import Scorer
from ally_fall.metrics import binary
from ally_fall.posture import POSTURES, posture_map

logger = logging.getLogger(__name__)

BAR: float = 0.80
BAR_POSTURES: tuple[str, ...] = ("upright", "sitting", "lying")
SEEDS: tuple[int, ...] = (0, 1, 2)
MISSING_BINS: tuple[tuple[str, float], ...] = (
    ("no ticks lost", 0.0),
    ("up to a fifth of the ticks", 0.2),
    ("a fifth to half of the ticks", 0.5),
    ("over half of the ticks", 1.0),
)


@dataclass(frozen=True)
class PostureScore:
    posture: str
    windows: int  # windows whose true posture this is
    recall: float
    precision: float
    f1: float


@dataclass(frozen=True)
class SeedResult:
    """One seed: the confusion matrix (true rows, predicted columns) and mean bar-posture F1 of
    each slice (person, dataset, share of ticks with no pose)."""

    seed: int
    matrix: NDArray[np.int64]
    scores: list[PostureScore]
    by_person: dict[str, float]
    by_dataset: dict[str, float]
    by_missing: dict[str, float]


def true_postures(labels: NDArray[np.int64], names: tuple[str, ...]) -> NDArray[np.int64]:
    """The posture index of each window's true OmniFall label."""
    return posture_map(names)[labels]


def confusion(pred: NDArray[np.int64], true: NDArray[np.int64]) -> NDArray[np.int64]:
    matrix = np.zeros((len(POSTURES), len(POSTURES)), dtype=np.int64)
    np.add.at(matrix, (true, pred), 1)
    return matrix


def posture_scores(pred: NDArray[np.int64], true: NDArray[np.int64]) -> list[PostureScore]:
    scores: list[PostureScore] = []
    for index, posture in enumerate(POSTURES):
        counts = binary(pred == index, true == index)
        scores.append(
            PostureScore(posture, int((true == index).sum()), counts.recall, counts.precision,
                         counts.f1)
        )
    return scores


def bar_met(f1_by_seed: Sequence[Mapping[str, float]]) -> bool:
    """True when upright, sitting and lying reach the bar on every seed (and there is a seed)."""
    return bool(f1_by_seed) and all(
        f1[posture] >= BAR for f1 in f1_by_seed for posture in BAR_POSTURES
    )


def missing_share(split: BuiltSplit, ticks: int, chunk: int = 4096) -> NDArray[np.float64]:
    """Per window: the share of its ticks in which no pose was found (all visibilities zero)."""
    parts: list[NDArray[np.float64]] = []
    for start in range(0, len(split.window_end), chunk):
        end = split.window_end[start:start + chunk]
        windows = window_features(split.features, end, ticks)  # (n, ticks, 33, 3)
        parts.append((windows[..., 2].max(axis=-1) == 0).mean(axis=1))
    return np.concatenate(parts) if parts else np.zeros(0)


def _bar_mean(pred: NDArray[np.int64], true: NDArray[np.int64]) -> float | None:
    """Mean F1 of the bar postures that occur in this slice; None when none does."""
    f1 = [s.f1 for s in posture_scores(pred, true) if s.posture in BAR_POSTURES and s.windows]
    return float(np.mean(f1)) if f1 else None


def _slices(
    keys: NDArray[np.str_], pred: NDArray[np.int64], true: NDArray[np.int64]
) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in sorted(set(keys.tolist())):
        value = _bar_mean(pred[keys == key], true[keys == key])
        if value is not None:
            out[key] = value
    return out


def predict_postures(
    scorer: Scorer, split: BuiltSplit, chunk: int = 2048
) -> NDArray[np.int64]:
    top: list[NDArray[np.int64]] = []
    total = len(split.window_end)
    for start in range(0, total, chunk):
        index = np.arange(start, min(start + chunk, total), dtype=np.int64)
        top.append(scorer.top_posture_batch(window_tensor(split, index, scorer.window_ticks)))
    return np.concatenate(top) if top else np.zeros(0, dtype=np.int64)


def evaluate(
    seed: int,
    scorer: Scorer,
    split: BuiltSplit,
    names: tuple[str, ...],
) -> SeedResult:
    pred = predict_postures(scorer, split)
    true = true_postures(split.y, names)
    video = window_video(split)
    dataset = split.dataset[video]
    person = np.array([f"{d}/{s}" for d, s in zip(dataset.tolist(), split.subject[video].tolist(),
                                                  strict=True)])
    missing = missing_share(split, scorer.window_ticks)
    bins = np.array([_missing_label(m) for m in missing])
    return SeedResult(
        seed,
        confusion(pred, true),
        posture_scores(pred, true),
        _slices(person, pred, true),
        _slices(dataset.astype(str), pred, true),
        _slices(bins, pred, true),
    )


def _missing_label(share: float) -> str:
    if share == 0.0:
        return MISSING_BINS[0][0]
    for label, upper in MISSING_BINS[1:]:
        if share <= upper:
            return label
    return MISSING_BINS[-1][0]


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def report_markdown(results: Sequence[SeedResult]) -> str:
    f1_by_seed = [{s.posture: s.f1 for s in r.scores} for r in results]
    met = bar_met(f1_by_seed)
    verdict = (
        f"**The bar is met:** upright, sitting and lying reach F1 {BAR:.2f} or more on every seed."
        if met
        else f"**The bar is missed:** upright, sitting and lying do not all reach F1 {BAR:.2f} "
        "on every seed (table below). No tuning was done; the next step is the PO's decision "
        "(D-72: a `posture` label set and a retrain, or leave it)."
    )
    below = sorted(
        {d for r in results for d, f1 in r.by_dataset.items() if f1 < BAR}
    )
    caveat = (
        f"The bar is judged on the pooled windows, which are mostly from one dataset. "
        f"Datasets below the bar (mean F1 of upright, sitting and lying under {BAR:.2f} on at "
        f"least one seed): {', '.join(below)}."
        if below
        else ""
    )
    lines = [
        "# Posture of the fall model, validation split (P1, D-72)",
        "",
        "The fall model is unchanged; its 16 class probabilities are added up in five postures "
        "(`ally_fall.posture`). **Validation split only: the test split was not opened.** "
        "Seeds " + ", ".join(str(r.seed) for r in results) + " of the shipped TCN (float).",
        "",
        verdict,
        "",
        *([caveat, ""] if caveat else []),
        "## F1 per posture and seed",
        "",
        _row(["posture", "true windows (seed 0)"] + [f"F1 seed {r.seed}" for r in results]),
        _row(["---"] * (2 + len(results))),
    ]
    for index, posture in enumerate(POSTURES):
        windows = results[0].scores[index].windows if results else 0
        f1 = [f"{r.scores[index].f1:.3f}" for r in results]
        mark = " (bar)" if posture in BAR_POSTURES else ""
        lines.append(_row([posture + mark, str(windows)] + f1))
    for r in results:
        lines += ["", f"## Confusion matrix, seed {r.seed} (true rows, predicted columns)", "",
                  _row(["true \\ predicted"] + list(POSTURES)), _row(["---"] * (1 + len(POSTURES)))]
        for index, posture in enumerate(POSTURES):
            lines.append(_row([posture] + [str(int(v)) for v in r.matrix[index]]))
    for title, attribute in (("dataset", "by_dataset"), ("person", "by_person"),
                             ("share of ticks with no pose", "by_missing")):
        keys = sorted({k for r in results for k in getattr(r, attribute)})
        lines += ["", f"## Mean F1 of upright, sitting and lying, by {title}", "",
                  _row([title] + [f"seed {r.seed}" for r in results]),
                  _row(["---"] * (1 + len(results)))]
        for key in keys:
            cells = [f"{getattr(r, attribute)[key]:.3f}" if key in getattr(r, attribute) else ""
                     for r in results]
            lines.append(_row([key] + cells))
    lines += [
        "",
        "## Limits (state wherever this is shown)",
        "",
        "- Validation only: young staged actors in four datasets, no older adults.",
        *(
            [f"- The validation split has {len(results[0].by_person)} people in "
             f"{len(results[0].by_dataset)} datasets, so each dataset figure is one person."]
            if results and len(results[0].by_person) == len(results[0].by_dataset)
            else []
        ),
        "- Floor against bed or sofa is not tested (D-51 risk (1) stays open).",
        "- Kneeling, squatting, crawl and jump have no training windows, so four of five "
        "postures can be recognised; they fall into \"other\".",
        "- `lying` includes `lie_down`, `fallen` and `fall`: the model cannot tell lying from "
        "fallen reliably (F3b), so they are one posture here.",
        "- Phone speed and the INT8 file are not part of this report.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    from ally_fall.train import _splits, load_config

    parser = argparse.ArgumentParser(description="Posture of the fall model on validation.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, default=Path("configs/f2-grid"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    for path in (args.data_root, args.config_dir, args.out):
        refuse_test_split(path)
    results: list[SeedResult] = []
    for seed in SEEDS:
        config = args.config_dir / f"tcn-omnifall16-b0.999-s{seed}.toml"
        cfg = load_config(config)
        weights = args.data_root / "runs" / cfg.run_id / "best.pt"
        scorer = Scorer.load(config, weights, args.data_root / "omnifall")
        names, _, val = _splits(args.data_root, cfg.pose_model, cfg.label_set)
        logger.info("%s: %d validation windows", cfg.run_id, len(val.window_end))
        results.append(evaluate(seed, scorer, val, names))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report_markdown(results), encoding="utf-8")
    print(f"posture report written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
