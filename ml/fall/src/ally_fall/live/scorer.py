"""Window in, fall score out, from a saved run (F3c Task 3, D-69).

The score is the probability of the fall class, fall plus fallen (D-55), the same class the
validation numbers are measured on. The run's config decides the model, the label set, the
window length and whether windows are hip-centred, so a swap for the final model is one path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from ally_fall.keypoints import POINTS
from ally_fall.labels import class_names, fall_classes
from ally_fall.models import build_model
from ally_fall.omnifall import label_names
from ally_fall.train import RunConfig, load_config, model_input


class Scorer:
    """A loaded fall model; `score` for one window, `score_batch` for many."""

    def __init__(self, model: torch.nn.Module, cfg: RunConfig, fall: tuple[int, ...]) -> None:
        self._model = model.eval()
        self._cfg = cfg
        self._fall = list(fall)

    @classmethod
    def load(cls, config: Path, weights: Path, omnifall_dir: Path) -> Scorer:
        """Load a run: its TOML config, its `best.pt`, and OmniFall's `labels/label2id.csv` folder.

        Raises FileNotFoundError for a missing file and ValueError when the weights do not fit
        the config (for instance a groups5 model with an omnifall16 config).
        """
        cfg = load_config(config)
        names = label_names(omnifall_dir)
        classes = class_names(cfg.label_set, names)
        model = build_model(cfg.model, len(classes), cfg.hidden, None, cfg.window_ticks)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        try:
            model.load_state_dict(state)
        except RuntimeError as error:
            raise ValueError(
                f"{weights} does not match the config {config} "
                f"({cfg.model}, {cfg.label_set}, {cfg.window_ticks} ticks): {error}"
            ) from error
        return cls(model, cfg, fall_classes(cfg.label_set, names))

    @property
    def window_ticks(self) -> int:
        return self._cfg.window_ticks

    @property
    def run_id(self) -> str:
        return self._cfg.run_id

    def _probabilities(self, windows: NDArray[np.float32]) -> torch.Tensor:
        expected = (3, self.window_ticks, POINTS)
        if windows.ndim != 4 or windows.shape[1:] != expected:
            raise ValueError(f"expected windows of shape (n, {expected}), got {windows.shape}")
        x = torch.from_numpy(np.ascontiguousarray(model_input(windows, self._cfg)))
        with torch.no_grad():
            return torch.softmax(self._model(x), dim=1)

    def score_batch(self, windows: NDArray[np.float32]) -> NDArray[np.float32]:
        """Fall scores (0 to 1) for windows of shape (n, 3, window_ticks, 33)."""
        probs = self._probabilities(windows)
        return probs[:, self._fall].sum(dim=1).numpy().astype(np.float32)

    def top_class_batch(self, windows: NDArray[np.float32]) -> NDArray[np.int64]:
        """The most likely class of each window (the rule F2 and F3b score by)."""
        return self._probabilities(windows).argmax(dim=1).numpy().astype(np.int64)

    def top_is_fall_batch(self, windows: NDArray[np.float32]) -> NDArray[np.bool_]:
        """True where the most likely class is fall or fallen."""
        return np.isin(self.top_class_batch(windows), self._fall)

    def score(self, window: NDArray[np.float32]) -> float:
        """The fall score of one window of shape (3, window_ticks, 33)."""
        expected = (3, self.window_ticks, POINTS)
        if window.shape != expected:
            raise ValueError(f"expected a window of shape {expected}, got {window.shape}")
        return float(self.score_batch(window[None])[0])
