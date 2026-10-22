"""Class-balanced loss weights from the effective number of samples (D-55, D-62).

weight_c = (1 - beta) / (1 - beta ** n_c): the inverse of the effective number
(1 - beta ** n) / (1 - beta). A class with no training windows gets weight 0 (it cannot be
learned, and its effective number is 0). The weights of the present classes average 1.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def class_weights(counts: NDArray[np.int64], beta: float) -> NDArray[np.float32]:
    if not 0.0 <= beta < 1.0:
        raise ValueError(f"beta must be in [0, 1), got {beta}")
    counts = np.asarray(counts, dtype=np.float64)
    present = counts > 0
    weights = np.zeros(len(counts), dtype=np.float64)
    weights[present] = (1.0 - beta) / (1.0 - np.power(beta, counts[present]))
    weights[present] /= weights[present].mean()
    return weights.astype(np.float32)
