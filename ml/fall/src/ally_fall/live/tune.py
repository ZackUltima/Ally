"""Choosing the alert threshold and run length on validation windows only (F3c Task 4, D-69).

The rule for the default is fixed here, before the numbers are looked at: keep every setting
that finds at least (best falls found - 1) of the validation falls, take the one with the fewest
false alerts, and break ties by the shorter run, then the lower threshold. Falls and false
alerts are counted as in F2 and F3b (`metrics.count_events`: one alert per 30 s, a fall is found
if an alert starts inside its labelled run). The test split is never read here.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ally_fall.data import BuiltSplit
from ally_fall.live.alert import active_mask
from ally_fall.live.replay import window_groups
from ally_fall.metrics import count_events
from ally_fall.ticks import TICK_S

THRESHOLDS: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
REQUIRED: tuple[int, ...] = (1, 2, 3, 5, 8, 10)


@dataclass(frozen=True)
class GridRow:
    threshold: float
    required: int
    detected: int
    missed: int
    false_alerts: int
    false_per_hour: float


def alert_grid(
    split: BuiltSplit,
    scores: NDArray[np.float32],
    fall: tuple[int, int],
    thresholds: Sequence[float] = THRESHOLDS,
    requireds: Sequence[int] = REQUIRED,
) -> list[GridRow]:
    """Falls found and false alerts of every (threshold, run length) on the split's videos."""
    groups = window_groups(split)
    hours = len(split.window_end) * TICK_S / 3600.0
    rows: list[GridRow] = []
    for threshold in thresholds:
        for required in requireds:
            detected = missed = false_alerts = 0
            for group in groups:
                video_start = split.video_start[group.video]
                times = (split.window_end[group.index] - video_start) * TICK_S
                events = count_events(
                    times,
                    active_mask(scores[group.index], threshold, required),
                    split.y[group.index],
                    fall[0],
                    fall[1],
                )
                detected += events.detected
                missed += events.missed
                false_alerts += events.false_alerts
            rows.append(
                GridRow(threshold, required, detected, missed, false_alerts, false_alerts / hours)
            )
    return rows


def choose_default(rows: Sequence[GridRow]) -> tuple[float, int]:
    """The default (threshold, run length) by the rule in the module note."""
    best = max(row.detected for row in rows)
    kept = [row for row in rows if row.detected >= best - 1]
    chosen = min(kept, key=lambda r: (r.false_alerts, r.required, r.threshold))
    return chosen.threshold, chosen.required


def grid_markdown(
    rows: Sequence[GridRow], run_id: str, chosen: tuple[float, int], hours: float
) -> str:
    lines = [
        f"# Alert values for the live demo: {run_id}",
        "",
        "Computed on the **validation** windows only (the test split is not read). "
        f"{hours:.2f} h of video windows. Each row is an alert rule: the fall score at or above "
        "the threshold on N ticks in a row (one tick is 0.1 s, so N adds N x 0.1 s). Falls and "
        "false alerts are counted as in F2 and F3b (one alert per 30 s; a fall is found if an "
        "alert starts inside its labelled run).",
        "",
        "Rule for the default, fixed before the numbers were looked at: keep every row that finds "
        "at least (best falls found - 1) falls, take the fewest false alerts, then the shorter N, "
        "then the lower threshold. The chosen row is in bold.",
        "",
        "Read with care: the validation set is only a few minutes of short staged clips from 4 "
        "people, mostly falls and the movements around them. \"False alerts per hour\" is "
        "therefore not a rate for ordinary days at home, and a one-tick run (N = 1) is "
        "unprotected against the pose noise of a live camera, which these clean clips do not "
        "show. The numbers choose a starting point; they are not a performance claim.",
        "",
        "| Threshold | N | Falls found | Missed | False alerts | False alerts per hour |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        mark = (row.threshold, row.required) == chosen
        threshold = f"**{row.threshold:.2f}**" if mark else f"{row.threshold:.2f}"
        required = f"**{row.required}**" if mark else f"{row.required}"
        lines.append(
            f"| {threshold} | {required} | {row.detected} | {row.missed} | {row.false_alerts} | "
            f"{row.false_per_hour:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    from ally_fall.labels import fall_classes
    from ally_fall.live.replay import score_split
    from ally_fall.live.scorer import Scorer
    from ally_fall.train import _splits, load_config

    parser = argparse.ArgumentParser(description="Choose the live alert values on validation.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    scorer = Scorer.load(args.config, args.weights, args.data_root / "omnifall")
    names, _, val = _splits(args.data_root, cfg.pose_model, cfg.label_set)
    fall = fall_classes(cfg.label_set, names)
    scores, _ = score_split(val, scorer)
    rows = alert_grid(val, scores, (fall[0], fall[1]))
    chosen = choose_default(rows)
    hours = len(val.window_end) * TICK_S / 3600.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(grid_markdown(rows, cfg.run_id, chosen, hours), encoding="utf-8")
    print(f"default threshold {chosen[0]:.2f}, N {chosen[1]}; table written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
