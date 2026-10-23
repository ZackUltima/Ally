"""The pretrained ST-GCN candidate (D-63, F2 Task 9): MMAction2's ST-GCN, joint stream, 2D COCO-17,
trained on NTU RGB+D 120 (ml/fall/results/checks/pretrained-stgcn.md).

The forward pass is rebuilt in plain PyTorch (no MMAction2 or mmcv dependency); module names
follow the checkpoint, so its weights load by name. Training uses the published form (einsum over
A * PA); `export_form` folds it into the form that converts to INT8 (check 4, runs 3-5). The
input stays (B, 3, 30, 33), MediaPipe order; `coco17_input` picks the 17 COCO points by name and
maps visible x, y to the checkpoint's [-1, 1] range. MMAction2 code is Apache 2.0 (OpenMMLab);
the weights derive from NTU RGB+D (academic, non-commercial; not cleared for PIXEL, D-36).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from ally_fall.windows import WINDOW_TICKS

# MediaPipe Pose Landmarker's 33 landmarks, in index order (Google's landmark guide).
MEDIAPIPE_NAMES: tuple[str, ...] = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky",
    "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
)
# COCO's 17 keypoints, in index order (MMPose's COCO metadata).
COCO17_NAMES: tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
# MMAction2 graph layout "coco": the 16 bones pointing towards the centre (node 0, the nose).
COCO_INWARD: tuple[tuple[int, int], ...] = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6), (9, 7), (7, 5), (10, 8), (8, 6),
    (5, 0), (6, 0), (1, 0), (3, 1), (2, 0), (4, 2),
)
T, V, C, K = WINDOW_TICKS, len(COCO17_NAMES), 3, 3  # ticks, points, (x, y, score), graph subsets
# (cin, cout, stride, residual) of the 10 blocks: MMAction2 defaults (base 64, x2 at stages 5, 8).
BLOCKS: tuple[tuple[int, int, int, str], ...] = (
    (3, 64, 1, "none"), (64, 64, 1, "id"), (64, 64, 1, "id"), (64, 64, 1, "id"),
    (64, 128, 2, "conv"), (128, 128, 1, "id"), (128, 128, 1, "id"),
    (128, 256, 2, "conv"), (256, 256, 1, "id"), (256, 256, 1, "id"),
)
T_OUT = (((T + 1) // 2) + 1) // 2  # ticks left after the two stride-2 stages (30 -> 15 -> 8)


def coco17_from_mediapipe() -> tuple[int, ...]:
    """The MediaPipe index of each COCO-17 point, looked up by name (check 3)."""
    return tuple(MEDIAPIPE_NAMES.index(name) for name in COCO17_NAMES)


_COCO17 = coco17_from_mediapipe()


def coco17_input(x: torch.Tensor) -> torch.Tensor:
    """(B, 3, T, 33) MediaPipe x, y in [0, 1] and visibility -> (B, 3, T, 17) for the checkpoint:
    visible x, y become 2p - 1 (MMAction2's PreNormalize2D); a missing point stays (0, 0, 0)."""
    p = x[..., list(_COCO17)]
    seen = (p[:, 2:3] > 0).to(p.dtype)
    return torch.cat([(p[:, :2] * 2.0 - 1.0) * seen, p[:, 2:3]], dim=1)


def coco_spatial_graph() -> torch.Tensor:
    """MMAction2's Graph(layout="coco", mode="stgcn_spatial"), max_hop 1: subsets self links,
    links towards the centre, links away from it; each column normalised by its degree."""
    adj = np.eye(V)
    for i, j in COCO_INWARD:
        adj[i, j] = adj[j, i] = 1.0
    hop = np.where(np.eye(V) > 0, 0.0, np.where(adj > 0, 1.0, np.inf))  # distances up to 1 hop
    norm = adj / adj.sum(axis=0, keepdims=True)
    centre = 0
    a = np.zeros((K, V, V))
    for j in range(V):
        for i in range(V):
            if hop[j, i] == 0:
                a[0, j, i] = norm[j, i]
            elif hop[j, i] == 1:
                closer = hop[j, centre] >= hop[i, centre]
                a[1 if closer else 2, j, i] = norm[j, i]
    return torch.from_numpy(a.astype(np.float32))


class Tcn(nn.Module):
    """MMAction2 unit_tcn: a (k, 1) convolution over time, then batch norm."""

    def __init__(self, cin: int, cout: int, k: int = 9, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, (k, 1), padding=((k - 1) // 2, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class Gcn(nn.Module):
    """MMAction2 unit_gcn, adaptive="importance": a 1x1 conv to K subsets, then each subset mixed
    over the joints by A * PA (A fixed, PA learned), summed, batch norm, ReLU."""

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.register_buffer("A", coco_spatial_graph())
        self.PA = nn.Parameter(torch.ones(K, V, V))
        self.conv = nn.Conv2d(cin, cout * K, 1)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, t, v = x.shape
        y = self.conv(x).view(n, K, -1, t, v)
        y = torch.einsum("nkctv,kvw->nctw", y, self.A * self.PA).contiguous()
        return self.act(self.bn(y))


class GcnMixed(nn.Module):
    """The same graph step in the form that converts to INT8 (check 4 run 5): the K subsets joined
    on the joint axis and mixed and summed by one Linear(K*V -> V), so no plain ADD and no batch
    norm after a sum stays in float; the batch norm is folded into the conv and a shift."""

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.cout = cout
        self.conv = nn.Conv2d(cin, cout * K, 1)
        self.mix = nn.Linear(K * V, V, bias=False)
        self.shift = nn.Parameter(torch.zeros(1, cout, 1, 1))
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, t, v = x.shape
        y = self.conv(x).reshape(b, K, self.cout * t, v).permute(0, 2, 1, 3)
        y = self.mix(y.reshape(b, self.cout * t, K * v)).reshape(b, self.cout, t, v)
        return self.act(y + self.shift)


class Block(nn.Module):
    """One ST-GCN block: graph step, time convolution, residual, ReLU."""

    def __init__(self, cin: int, cout: int, stride: int, residual: str, mobile: bool) -> None:
        super().__init__()
        self.kind = residual
        self.gcn: nn.Module = GcnMixed(cin, cout) if mobile else Gcn(cin, cout)
        self.tcn = Tcn(cout, cout, 9, stride)
        if residual == "conv":
            self.residual = Tcn(cin, cout, 1, stride)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.tcn(self.gcn(x))
        if self.kind == "id":
            y = y + x
        elif self.kind == "conv":
            y = y + self.residual(x)
        return self.relu(y)


def _backbone(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    for block in model.gcn:
        x = block(x)
    return model.fc(model.pool(x).flatten(1))


class PretrainedSTGCN(nn.Module):
    """Input (B, 3, 30, 33), MediaPipe order. `checkpoint` None gives the random start (the
    control run, F2 decision 7); otherwise every layer but the head loads by name."""

    def __init__(self, n_classes: int, checkpoint: Path | None = None) -> None:
        super().__init__()
        self.data_bn = nn.BatchNorm1d(V * C)  # one person (MMAction2 data_bn_type "VC")
        self.gcn = nn.ModuleList(Block(*b, mobile=False) for b in BLOCKS)
        self.pool = nn.AvgPool2d((T_OUT, V))  # fixed size, stays INT8 (D-53 rule 2)
        self.fc = nn.Linear(BLOCKS[-1][1], n_classes)
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

    def load_checkpoint(self, path: Path) -> None:
        """Load an MMAction2 checkpoint (backbone.*, cls_head.*); the head stays new."""
        raw = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
        state = {k.removeprefix("backbone.").removeprefix("cls_head."): v for k, v in raw.items()}
        state = {k: v for k, v in state.items() if not k.startswith("fc.")}
        missing, unexpected = self.load_state_dict(state, strict=False)  # wrong shapes raise
        if unexpected or set(missing) != {"fc.weight", "fc.bias"}:
            raise RuntimeError(
                f"{path}: missing {sorted(missing)}, unexpected {sorted(unexpected)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = coco17_input(x)
        b = x.shape[0]
        x = x.permute(0, 3, 1, 2).reshape(b, V * C, T)  # channel = point * C + c, as MMAction2
        x = self.data_bn(x).reshape(b, V, C, T).permute(0, 2, 3, 1)
        return _backbone(self, x)


class MobileSTGCN(nn.Module):
    """`export_form` of a PretrainedSTGCN. Input (B, 3, 30, 17), already through `coco17_input`
    (on the phone, the 17-point selection and rescaling happen before the model)."""

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        # data_bn after training: one scale and shift per (point, channel), as a grouped 1x1 conv,
        # which the quantiser keeps in INT8 (a lone BatchNorm1d stays float, check 4 run 2).
        self.data_affine = nn.Conv2d(V * C, V * C, 1, groups=V * C)
        self.gcn = nn.ModuleList(Block(*b, mobile=True) for b in BLOCKS)
        self.pool = nn.AvgPool2d((T_OUT, V))
        self.fc = nn.Linear(BLOCKS[-1][1], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = x.permute(0, 3, 1, 2).reshape(b, V * C, T, 1)
        x = self.data_affine(x).reshape(b, V, C, T).permute(0, 2, 3, 1)
        return _backbone(self, x)


def _bn_scale(bn: nn.BatchNorm1d | nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return scale, bn.bias - bn.running_mean * scale


@torch.no_grad()
def export_form(model: PretrainedSTGCN) -> MobileSTGCN:
    """The trained model, rewritten exactly into the form that converts to INT8: the input batch
    norm as a grouped conv, each graph step's batch norm folded into its conv (the per-channel
    scale passes through the joint mix), and A * PA as the fixed weight of one Linear."""
    mobile = MobileSTGCN(model.fc.out_features)
    mobile.load_state_dict({k: v for k, v in model.state_dict().items()
                            if not k.startswith("data_bn") and ".gcn." not in k}, strict=False)
    scale, shift = _bn_scale(model.data_bn)
    mobile.data_affine.weight.copy_(scale.view(-1, 1, 1, 1))
    mobile.data_affine.bias.copy_(shift)
    for src, dst in zip(model.gcn, mobile.gcn, strict=True):
        g = src.gcn
        scale, shift = _bn_scale(g.bn)
        dst.gcn.conv.weight.copy_(g.conv.weight * scale.repeat(K).view(-1, 1, 1, 1))
        dst.gcn.conv.bias.copy_(g.conv.bias * scale.repeat(K))
        dst.gcn.mix.weight.copy_((g.A * g.PA).permute(2, 0, 1).reshape(V, K * V))
        dst.gcn.shift.copy_(shift.view(1, -1, 1, 1))
    return mobile.eval()
