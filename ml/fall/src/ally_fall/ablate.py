"""The F3b ablations: nine changed settings x three seeds on the chosen model (D-55 item 8, D-68).

`configs` writes the 27 run configs from the chosen model's F2 configs; `summary` compares the
finished runs with that F2 setting and writes `ablations.md`. Validation only: a path under the
test split is refused. A setting is called "no clear difference" when its mean F1 differs from the
F2 setting's by no more than the setting's own seed range (the F2 rule); a setting that is clearly
better is flagged so the PO decides before anything changes (Task 4).

Usage:
  python -m ally_fall.ablate configs [--results-dir results/runs] [--configs-dir configs/f3b-ablate]
  python -m ally_fall.ablate summary [--results-dir results/runs] [--runs-dir results/f3b/runs]
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from ally_fall.augment import AugmentConfig
from ally_fall.choose import POSE, refuse_test_split
from ally_fall.choose import read_validation as read_run
from ally_fall.runs import write_config
from ally_fall.summary import SEEDS
from ally_fall.train import RESULTS_DIR, RunConfig

MODEL = "tcn"  # the model chosen in F3b Task 1 (D-68, PO 2026-10-26)
LABEL_SET = "omnifall16"
BETA = 0.999
EPS = 1e-9
RUNS_DIR = Path(__file__).resolve().parents[2] / "results" / "f3b" / "runs"
OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "f3b"
CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs" / "f3b-ablate"
REPORT_NAME = "ablations.md"


@dataclass(frozen=True)
class Setting:
    """One changed setting: the config keys it changes (`augment.x` for an augmentation key)."""

    name: str
    description: str
    changes: dict[str, Any]


SETTINGS: tuple[Setting, ...] = (
    Setting("hip_centred", "input moved to the mean visible hip centre", {"hip_centred": True}),
    Setting("window_21", "2.1 s window (21 ticks at 10 fps) instead of 3 s", {"window_ticks": 21}),
    Setting("no_augment", "no augmentation at all",
            {"augment.mirror_p": 0.0, "augment.scale": 0.0, "augment.dropout_p": 0.0,
             "augment.edge_p": 0.0}),
    Setting("no_mirror", "mirror removed", {"augment.mirror_p": 0.0}),
    Setting("no_scale", "scale removed", {"augment.scale": 0.0}),
    Setting("no_dropout", "point dropout removed", {"augment.dropout_p": 0.0}),
    Setting("no_edge", "edge cut-off removed", {"augment.edge_p": 0.0}),
    Setting("rotation", "rotation of up to 0.1 rad added", {"augment.rotate": 0.1}),
    Setting("noise", "Gaussian noise (sd 0.01 of the picture) added", {"augment.noise_sd": 0.01}),
)


def flat(cfg: RunConfig) -> dict[str, Any]:
    """The config as one flat dict; augmentation keys are `augment.<name>`."""
    data = asdict(cfg)
    augment = data.pop("augment")
    return {**data, **{f"augment.{k}": v for k, v in augment.items()}}


def apply(cfg: RunConfig, setting: Setting) -> RunConfig:
    """The config with the setting's keys changed and a run ID that names the setting."""
    top = {k: v for k, v in setting.changes.items() if not k.startswith("augment.")}
    augment = {k.removeprefix("augment."): v for k, v in setting.changes.items()
               if k.startswith("augment.")}
    return replace(cfg, run_id=f"{cfg.run_id}-ab-{setting.name}", **top,
                   augment=replace(cfg.augment, **augment))


def setting_of(cfg: RunConfig, base: RunConfig) -> Setting | None:
    """Which setting `cfg` is, found from how it differs from `base` and never from its ID."""
    a, b = flat(cfg), flat(base)
    diff = {k: v for k, v in a.items() if k != "run_id" and v != b[k]}
    return next((s for s in SETTINGS if s.changes == diff), None)


def ablation_configs(bases: Sequence[RunConfig]) -> list[RunConfig]:
    return [apply(base, setting) for setting in SETTINGS for base in bases]


def write_configs(configs: Sequence[RunConfig], configs_dir: Path) -> list[Path]:
    paths = [configs_dir / f"{cfg.run_id}.toml" for cfg in configs]
    for cfg, path in zip(configs, paths, strict=True):
        write_config(cfg, path)
    return paths


def _runs(directory: Path) -> list[dict[str, Any]]:
    refuse_test_split(directory)
    runs = [read_run(p) for p in sorted(directory.glob("*.json"))]
    return [r for r in runs if isinstance(r, dict) and "config" in r]


def _config(raw: dict[str, Any]) -> RunConfig:
    values = {k: v for k, v in raw.items() if k != "augment"}
    return RunConfig(**values, augment=AugmentConfig(**raw.get("augment", {})))


def base_configs(results_dir: Path) -> list[RunConfig]:
    """The F2 configs of the chosen model, seeds 0-2, read from its results (never rebuilt)."""
    found = [_config(r["config"]) for r in _runs(results_dir)
             if r["config"]["model"] == MODEL and r["config"]["label_set"] == LABEL_SET
             and float(r["config"]["beta"]) == BETA and r["config"].get("pose_model", POSE) == POSE
             and not r["config"].get("checkpoint", "")]
    found.sort(key=lambda c: c.seed)
    if [c.seed for c in found] != list(SEEDS):
        raise ValueError(f"{MODEL}: needs one F2 run per seed {list(SEEDS)}, "
                         f"found {[c.seed for c in found]}")
    return found


@dataclass(frozen=True)
class Row:
    """One setting's finished runs and what they say against the F2 setting."""

    setting: str
    description: str
    f1: dict[int, float]
    found: dict[int, int]
    gap: float | None
    verdict: str

    @property
    def complete(self) -> bool:
        return len(self.f1) == len(SEEDS)

    @property
    def f1_range(self) -> float:
        return max(self.f1.values()) - min(self.f1.values()) if self.f1 else 0.0


def _mean(values: dict[int, float] | dict[int, int]) -> float:
    return sum(values.values()) / len(values)


def summarise(results_dir: Path, runs_dir: Path) -> list[Row]:
    bases = {c.seed: c for c in base_configs(results_dir)}
    base_runs = {r["config"]["seed"]: r for r in _runs(results_dir)
                 if _config(r["config"]) in bases.values()}
    base_f1 = _mean({s: float(r["val"]["f1"]) for s, r in base_runs.items()})
    rows: list[Row] = []
    for setting in SETTINGS:
        f1: dict[int, float] = {}
        found: dict[int, int] = {}
        for r in _runs(runs_dir):
            cfg = _config(r["config"])
            if cfg.seed in bases and setting_of(cfg, bases[cfg.seed]) == setting:
                f1[cfg.seed] = float(r["val"]["f1"])
                found[cfg.seed] = int(r["events"]["detected"])
        gap = _mean(f1) - base_f1 if f1 else None
        rows.append(Row(setting.name, setting.description, f1, found, gap, _verdict(f1, gap)))
    done = sorted((r for r in rows if r.complete), key=lambda r: -(r.gap or 0.0))
    return done + [r for r in rows if not r.complete]


def _verdict(f1: dict[int, float], gap: float | None) -> str:
    if gap is None:
        return "not run"
    if len(f1) < len(SEEDS):
        return "incomplete"
    if abs(gap) <= max(f1.values()) - min(f1.values()) + EPS:
        return "no clear difference"
    return "better" if gap > 0 else "worse"


def summary_markdown(results_dir: Path, runs_dir: Path) -> str:
    rows = summarise(results_dir, runs_dir)
    base = [r for r in _runs(results_dir) if _config(r["config"]) in base_configs(results_dir)]
    base_f1 = _mean({r["config"]["seed"]: float(r["val"]["f1"]) for r in base})
    base_found = _mean({r["config"]["seed"]: int(r["events"]["detected"]) for r in base})
    lines = [
        "# F3b ablations (validation only)",
        "",
        f"Chosen model: {MODEL}, {LABEL_SET}, lite pose; the F2 setting is the reference "
        f"(mean validation fall-class F1 {base_f1:.3f}, falls found {base_found:.1f} of 57). "
        "Each setting changes only its own keys, 3 seeds. A setting is \"no clear difference\" "
        "when its mean F1 differs from the F2 setting's by no more than its own seed range.",
        "",
        "| setting | change | seeds | F1 mean | F1 range | gap vs F2 | falls found | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        n = f"{len(r.f1)} of {len(SEEDS)}"
        if r.f1:
            lines.append(f"| {r.setting} | {r.description} | {n} | {_mean(r.f1):.3f} | "
                         f"{r.f1_range:.3f} | {r.gap:+.3f} | {_mean(r.found):.1f} | {r.verdict} |")
        else:
            lines.append(f"| {r.setting} | {r.description} | {n} | | | | | {r.verdict} |")
    better = [r for r in rows if r.verdict == "better"]
    lines.append("")
    if better:
        for r in better:
            lines.append(f"- **{r.setting} BEATS the F2 setting** (gap {r.gap:+.3f} over a range "
                         f"of {r.f1_range:.3f}): stop and ask the PO before changing the final "
                         "configuration.")
    else:
        lines.append("- No finished setting is clearly better than the F2 setting.")
    return "\n".join([*lines, ""])


def write_summary(results_dir: Path, runs_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / REPORT_NAME
    path.write_text(summary_markdown(results_dir, runs_dir), encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("configs", help="write the 27 run configs")
    one.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    one.add_argument("--configs-dir", type=Path, default=CONFIGS_DIR)
    two = sub.add_parser("summary", help="compare the finished runs with the F2 setting")
    two.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    two.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    two.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    if args.command == "configs":
        paths = write_configs(ablation_configs(base_configs(args.results_dir)), args.configs_dir)
        print(f"wrote {len(paths)} configs to {args.configs_dir}")
    else:
        print(f"wrote {write_summary(args.results_dir, args.runs_dir, args.out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
