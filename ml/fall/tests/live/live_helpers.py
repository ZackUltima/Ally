"""Shared test helpers for the live-demo tests: a small saved run and a Scorer loaded from it."""

from pathlib import Path

import torch

from ally_fall.live.scorer import Scorer
from ally_fall.models import build_model

NAMES = (
    "walk", "fall", "fallen", "sit_down", "sitting", "lie_down", "lying", "stand_up",
    "standing", "other", "kneel_down", "kneeling", "squat_down", "squatting",
    "crawl", "jump",
)


def write_omnifall(root: Path) -> Path:
    labels = root / "omnifall" / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(f"{name},{i}" for i, name in enumerate(NAMES))
    (labels / "label2id.csv").write_text("label,id\n" + rows + "\n", encoding="utf-8")
    return root / "omnifall"


def make_scorer(root: Path, seed: int = 0) -> Scorer:
    """A random-weight 8-unit TCN (30 ticks, 16 classes) saved under `root` and loaded back."""
    torch.manual_seed(seed)
    torch.save(build_model("tcn", 16, hidden=8).state_dict(), root / "best.pt")
    (root / "run.toml").write_text(
        'run_id = "t"\nmodel = "tcn"\nlabel_set = "omnifall16"\nbeta = 0.999\n'
        'pose_model = "lite"\nhidden = 8\n',
        encoding="utf-8",
    )
    return Scorer.load(root / "run.toml", root / "best.pt", write_omnifall(root))
