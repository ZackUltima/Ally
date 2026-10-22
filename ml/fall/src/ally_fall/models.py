"""The three fall-model candidates (D-13), built after the D-53 spike so they convert to INT8.

Input (B, 3, 30, 33): x, y, visibility; 30 ticks; 33 MediaPipe points. D-53 rules: end
with a fixed-size average pool, never an adaptive one; the GRU gets an explicit start
state and uses both of its outputs; the ST-GCN's graph step is a fixed linear map over the
joint axis (the quantiser has no matmul rule). The GRU is reported in float only (D-13).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import nn

ModelName = Literal["tcn", "gru", "stgcn"]
C, T, V = 3, 30, 33

# MediaPipe Pose connections between the 33 landmarks.
EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), (12, 14), (14, 16), (16, 18),
    (16, 20), (16, 22), (18, 20), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27),
    (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)


def norm_adjacency() -> torch.Tensor:
    """Skeleton graph with self-loops, normalised as D^-1/2 (A + I) D^-1/2."""
    a = np.eye(V, dtype=np.float32)
    for i, j in EDGES:
        a[i, j] = a[j, i] = 1.0
    d = a.sum(1) ** -0.5
    return torch.from_numpy((d[:, None] * a * d[None, :]).astype(np.float32))


class TCN(nn.Module):
    """The 99 numbers of each tick are channels; convolutions run over time only."""

    def __init__(self, n_classes: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C * V, hidden, (5, 1), padding=(2, 0)), nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.Conv2d(hidden, hidden, (5, 1), padding=(4, 0), dilation=(2, 1)),
            nn.BatchNorm2d(hidden), nn.ReLU(),
            nn.AvgPool2d((T, 1)),
        )
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 1, 3, 2).reshape(-1, C * V, T, 1)
        return self.fc(self.net(x).flatten(1))


class GRUNet(nn.Module):
    """One GRU layer reads the 30 ticks in order; logits from its last step."""

    def __init__(self, n_classes: int, hidden: int = 64) -> None:
        super().__init__()
        self.hidden = hidden
        self.gru = nn.GRU(C * V, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_classes)

    def head(self, out: torch.Tensor, h_n: torch.Tensor) -> torch.Tensor:
        # out[:, -1] and h_n[-1] are the same numbers for one layer; averaging keeps the value
        # and uses both outputs, which the converter's GRU rule requires (D-53).
        return self.fc((out[:, -1] + h_n[-1]) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3).reshape(-1, T, C * V)
        h0 = x.new_zeros(1, x.shape[0], self.hidden)
        out, h_n = self.gru(x, h0)
        return self.head(out, h_n)


class GraphStep(nn.Module):
    """Mixes each joint with its skeleton neighbours; fixed, not trained."""

    def __init__(self) -> None:
        super().__init__()
        self.mix = nn.Linear(V, V, bias=False)
        with torch.no_grad():
            self.mix.weight.copy_(norm_adjacency().T)
        self.mix.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mix(x)


class STBlock(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.graph = GraphStep()
        self.spatial = nn.Sequential(nn.Conv2d(cin, cout, 1), nn.BatchNorm2d(cout), nn.ReLU())
        self.temporal = nn.Sequential(nn.Conv2d(cout, cout, (5, 1), padding=(2, 0)),
                                      nn.BatchNorm2d(cout), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.temporal(self.spatial(self.graph(x)))


class STGCN(nn.Module):
    """Two graph-and-time blocks, a fixed-size pool and a classifier."""

    def __init__(self, n_classes: int, hidden: int = 64) -> None:
        super().__init__()
        self.blocks = nn.Sequential(STBlock(C, hidden // 2), STBlock(hidden // 2, hidden),
                                    nn.AvgPool2d((T, V)))
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.blocks(x).flatten(1))


def build_model(name: ModelName, n_classes: int, hidden: int = 64) -> nn.Module:
    models: dict[str, type[TCN] | type[GRUNet] | type[STGCN]] = {
        "tcn": TCN, "gru": GRUNet, "stgcn": STGCN}
    if name not in models:
        raise ValueError(f"unknown model {name!r}")
    return models[name](n_classes, hidden)
