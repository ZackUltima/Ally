"""The F2 run plan as configs, and the overnight sequence (F2 Task 7 Step 5, D-62).

Decisions 2-8 of the F2 plan: the TCN on omnifall16, seed 0, with beta 0.99, 0.999 and 0.9999;
the beta with the best validation fall-class F1 is kept for every model; then TCN, GRU, light
ST-GCN and the pretrained ST-GCN x omnifall16 and groups5 x seeds 0-2, all on lite poses. The
control run and the full-pose re-runs depend on these results, so `after` configures them with the
rules fixed in D-62 (`summary.py`); every step ends by writing the F2 summary.

Usage: python -m ally_fall.runs overnight|after|summary --data-root ~/ally-data
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path

from ally_fall.augment import AugmentConfig
from ally_fall.labels import LabelSet
from ally_fall.models import ModelName
from ally_fall.summary import (
    SEEDS,
    Key,
    best_configuration,
    control_label_set,
    group_results,
    load_results,
    write_summary,
)
from ally_fall.train import RESULTS_DIR, RunConfig, run_queue
from ally_fall.train import main as train_main

logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
CHECKPOINT = ("models/stgcn_pt/"
              "stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-2d_20221129-612416c6.pth")
BETAS: tuple[float, ...] = (0.99, 0.999, 0.9999)
MODELS: tuple[ModelName, ...] = ("tcn", "gru", "stgcn", "stgcn_pt")
LABEL_SETS: tuple[LabelSet, ...] = ("omnifall16", "groups5")
BACKBONE_LR = 1e-4  # decision 5: pretrained layers at 1e-4, the new head at the usual 1e-3


def run_id(model: str, label_set: str, beta: float, seed: int, *, random_start: bool = False,
           pose_model: str = "lite") -> str:
    """The grid's IDs carry no suffix; the control (`-rand`) and the full-pose re-runs
    (`-full`) get one, so the queue never takes them for a finished grid run."""
    rand = "-rand" if random_start else ""
    pose = "" if pose_model == "lite" else f"-{pose_model}"
    return f"{model}-{label_set}-b{beta:g}-s{seed}{rand}{pose}"


def _config(model: ModelName, label_set: LabelSet, beta: float, seed: int, *,
            pretrained: bool = True, pose_model: str = "lite") -> RunConfig:
    loaded = model == "stgcn_pt" and pretrained
    return RunConfig(run_id=run_id(model, label_set, beta, seed, pose_model=pose_model,
                                   random_start=model == "stgcn_pt" and not pretrained),
                     model=model, label_set=label_set, beta=beta, seed=seed,
                     pose_model=pose_model, checkpoint=CHECKPOINT if loaded else "",
                     backbone_lr=BACKBONE_LR if loaded else None)


def beta_configs() -> list[RunConfig]:
    """Decision 2: the TCN on omnifall16, seed 0, once per beta."""
    return [_config("tcn", "omnifall16", beta, 0) for beta in BETAS]


def grid_configs(beta: float) -> list[RunConfig]:
    """Decisions 3-5: four models x two label sets x three seeds at the chosen beta (24 runs;
    the TCN on omnifall16, seed 0, is the beta run itself, so the queue skips it)."""
    return [_config(m, s, beta, seed) for m in MODELS for s in LABEL_SETS for seed in SEEDS]


def control_configs(beta: float, label_set: LabelSet) -> list[RunConfig]:
    """Decision 7: the pretrained network from a random start, one learning rate (1e-3) for
    every layer, on the label set from rule 1 (D-62)."""
    return [_config("stgcn_pt", label_set, beta, seed, pretrained=False) for seed in SEEDS]


def full_configs(key: Key) -> list[RunConfig]:
    """Decision 8: the best configuration (rule 2, D-62) again on full poses, seeds 0-2."""
    return [_config(key.model, key.label_set, key.beta, seed,  # type: ignore[arg-type]
                    pretrained=key.pretrained, pose_model="full") for seed in SEEDS]


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)  # a TOML basic string
    return repr(value)


# Added in F3b: left out of a config while at their defaults, so F2 configs read as they did.
_OMIT_WHEN_DEFAULT: frozenset[str] = frozenset({"hip_centred", "window_ticks"})


def write_config(cfg: RunConfig, path: Path) -> None:
    """One TOML file per run (T4); unset optional values and the default augmentation omitted."""
    lines = [f"{f.name} = {_toml_value(getattr(cfg, f.name))}" for f in fields(cfg)
             if f.name != "augment" and getattr(cfg, f.name) is not None
             and not (f.name in _OMIT_WHEN_DEFAULT and getattr(cfg, f.name) == f.default)]
    if cfg.augment != AugmentConfig():  # the F3b ablations change it
        lines += ["", "[augment]", *(f"{f.name} = {_toml_value(getattr(cfg.augment, f.name))}"
                                     for f in fields(cfg.augment))]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _val_f1(results_dir: Path, rid: str) -> float:
    path = results_dir / f"{rid}.json"
    if not path.exists():
        raise FileNotFoundError(f"no results for {rid} ({path})")
    return float(json.loads(path.read_text(encoding="utf-8"))["val"]["f1"])


def pick_beta(results_dir: Path) -> float:
    """Decision 2: the beta with the best validation fall-class F1; a tie keeps the smaller."""
    scores = {cfg.beta: _val_f1(results_dir, cfg.run_id) for cfg in beta_configs()}
    return max(BETAS, key=lambda b: (scores[b], -b))


def _write_choice(results_dir: Path, beta: float) -> None:
    rows = [f"| {c.beta:g} | {c.run_id} | {_val_f1(results_dir, c.run_id):.3f} |"
            for c in beta_configs()]
    (results_dir / "beta-choice.md").write_text("\n".join([
        "# Beta choice (F2 decision 2)", "",
        "TCN, omnifall16, seed 0, lite; kept: the best validation fall-class F1 (a tie keeps the"
        " smaller beta).", "", "| beta | run | validation fall-class F1 |", "|---|---|---|", *rows,
        "", f"**Kept: beta = {beta:g}**", ""]), encoding="utf-8")


def run_baseline_once(data_root: Path, results_dir: Path, pose_model: str = "lite") -> int:
    if (results_dir / f"baseline-{pose_model}.md").exists():
        return 0
    return train_main(["--baseline", "--data-root", str(data_root), "--pose-model", pose_model,
                       "--results-dir", str(results_dir)])


def _beta_or_none(results_dir: Path) -> float | None:
    try:
        return pick_beta(results_dir)
    except FileNotFoundError:
        return None


def overnight(data_root: Path, results_dir: Path, configs_dir: Path) -> int:
    """Baseline, the three beta runs, the beta pick, then the 24-run grid. Every step can be
    repeated: finished runs are skipped and an interrupted run resumes."""
    results_dir.mkdir(parents=True, exist_ok=True)
    failed = run_baseline_once(data_root, results_dir)
    beta_dir = configs_dir / "f2-beta"
    for cfg in beta_configs():
        write_config(cfg, beta_dir / f"{cfg.run_id}.toml")
    if run_queue(beta_dir, data_root, results_dir):
        logger.error("a beta run failed; the grid is not started")
        return 1
    beta = pick_beta(results_dir)
    _write_choice(results_dir, beta)
    logger.info("beta %g kept for the grid", beta)
    for cfg in grid_configs(beta):
        write_config(cfg, configs_dir / "f2-grid" / f"{cfg.run_id}.toml")
    failed |= run_queue(configs_dir / "f2-grid", data_root, results_dir)
    write_summary(results_dir, beta)
    return 1 if failed else 0


def after(data_root: Path, results_dir: Path, configs_dir: Path) -> int:
    """After the grid: the 3 control runs (rule 1), then the full-pose baseline and the best
    configuration on full poses (rule 2), then the summary. Repeatable like `overnight`."""
    beta = pick_beta(results_dir)
    groups = group_results(load_results(results_dir))
    label_set = control_label_set(groups, beta)
    best = best_configuration(groups, beta)
    if label_set is None or best is None:
        logger.error("the grid is not finished (a configuration lacks seeds); nothing started")
        write_summary(results_dir, beta)
        return 1
    for cfg in control_configs(beta, label_set):
        write_config(cfg, configs_dir / "f2-control" / f"{cfg.run_id}.toml")
    failed = run_queue(configs_dir / "f2-control", data_root, results_dir)
    failed |= run_baseline_once(data_root, results_dir, "full")
    for cfg in full_configs(best.key):
        write_config(cfg, configs_dir / "f2-full" / f"{cfg.run_id}.toml")
    failed |= run_queue(configs_dir / "f2-full", data_root, results_dir)
    write_summary(results_dir, beta)
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run the F2 plan")
    parser.add_argument("step", choices=["overnight", "after", "summary"])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--configs-dir", type=Path, default=CONFIGS_DIR)
    args = parser.parse_args(argv)
    if args.step == "summary":
        path = write_summary(args.results_dir, _beta_or_none(args.results_dir))
        logger.info("summary written to %s", path)
        return 0
    step = overnight if args.step == "overnight" else after
    return step(args.data_root.expanduser(), args.results_dir, args.configs_dir)


if __name__ == "__main__":
    sys.exit(main())
