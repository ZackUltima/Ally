"""Tests for the pretrained ST-GCN candidate (F2 Task 9, D-63). Checkpoints here are made up."""

from pathlib import Path

import pytest
import torch
from torch import nn

from ally_fall.pretrained import (
    COCO17_NAMES,
    MEDIAPIPE_NAMES,
    PretrainedSTGCN,
    coco17_from_mediapipe,
    coco17_input,
    coco_spatial_graph,
    export_form,
)

# Check 3 of ml/fall/results/checks/pretrained-stgcn.md (Google's landmarks, MMPose's COCO names).
CHECK3 = {"nose": 0, "left_eye": 2, "right_eye": 5, "left_ear": 7, "right_ear": 8,
          "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14,
          "left_wrist": 15, "right_wrist": 16, "left_hip": 23, "right_hip": 24,
          "left_knee": 25, "right_knee": 26, "left_ankle": 27, "right_ankle": 28}


def _fake_checkpoint(path: Path, n_classes: int = 60) -> dict[str, torch.Tensor]:
    """A checkpoint in MMAction2's key layout (backbone.*, cls_head.*), random weights."""
    torch.manual_seed(1)
    model = PretrainedSTGCN(n_classes)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d | nn.BatchNorm2d):
            module.running_mean.uniform_(-0.5, 0.5)
            module.running_var.uniform_(0.5, 2.0)
    sd = {("cls_head." if k.startswith("fc.") else "backbone.") + k: v
          for k, v in model.state_dict().items()}
    torch.save({"state_dict": sd, "meta": {"note": "made up"}}, path)
    return sd


def test_coco17_mapping_by_name() -> None:
    idx = coco17_from_mediapipe()
    assert len(MEDIAPIPE_NAMES) == 33 and len(COCO17_NAMES) == 17
    assert dict(zip(COCO17_NAMES, idx, strict=True)) == CHECK3
    assert all(MEDIAPIPE_NAMES[i] == name for name, i in zip(COCO17_NAMES, idx, strict=True))


def test_missing_points_stay_zero() -> None:
    x = torch.zeros(1, 3, 30, 33)
    x[0, :, :, 11] = torch.tensor([0.75, 0.25, 0.9]).view(3, 1)  # left shoulder, visible
    x[0, :, :, 12] = torch.tensor([0.0, 0.0, 0.0]).view(3, 1)  # right shoulder, missing
    out = coco17_input(x)
    assert out.shape == (1, 3, 30, 17)
    assert torch.allclose(out[0, :, 0, 5], torch.tensor([0.5, -0.5, 0.9]))
    assert torch.equal(out[0, :, :, 6], torch.zeros(3, 30))


def test_graph_is_three_subsets_over_coco_edges() -> None:
    a = coco_spatial_graph()
    assert a.shape == (3, 17, 17)
    assert torch.count_nonzero(a[0] - torch.diag(torch.diagonal(a[0]))) == 0  # self links only
    links = (a[1] + a[2]) > 0
    assert torch.equal(links, links.T) and int(links.sum()) == 2 * 16  # 16 COCO bones
    assert torch.allclose(a.sum(dim=(0, 1)), torch.ones(17))  # each column sums to 1


@pytest.mark.parametrize("batch", [1, 7])
@pytest.mark.parametrize("n_classes", [16, 5])
def test_pretrained_output_shape(batch: int, n_classes: int) -> None:
    model = PretrainedSTGCN(n_classes).eval()
    assert model(torch.zeros(batch, 3, 30, 33)).shape == (batch, n_classes)


def test_pretrained_no_adaptive_pool() -> None:
    for model in (PretrainedSTGCN(5), export_form(PretrainedSTGCN(5).eval())):
        assert not any("Adaptive" in type(m).__name__ for m in model.modules())


def test_pretrained_exportable() -> None:
    torch.export.export(PretrainedSTGCN(5).eval(), (torch.zeros(2, 3, 30, 33),))


def test_checkpoint_loads_all_but_head(tmp_path: Path) -> None:
    sd = _fake_checkpoint(tmp_path / "ckpt.pth")
    model = PretrainedSTGCN(16, tmp_path / "ckpt.pth")
    own = model.state_dict()
    for key, value in sd.items():
        name = key.removeprefix("backbone.").removeprefix("cls_head.")
        if name.startswith("fc."):
            assert own[name].shape[0] == 16  # a new head for our labels
        else:
            assert torch.equal(own[name], value), name


def test_checkpoint_wrong_shape_raises(tmp_path: Path) -> None:
    sd = _fake_checkpoint(tmp_path / "ckpt.pth")
    sd["backbone.gcn.0.gcn.conv.weight"] = torch.zeros(1, 3, 1, 1)
    torch.save({"state_dict": sd}, tmp_path / "bad.pth")
    with pytest.raises(RuntimeError):
        PretrainedSTGCN(16, tmp_path / "bad.pth")


def test_checkpoint_unexpected_key_raises(tmp_path: Path) -> None:
    sd = _fake_checkpoint(tmp_path / "ckpt.pth")
    sd["backbone.extra.weight"] = torch.zeros(1)
    torch.save({"state_dict": sd}, tmp_path / "bad.pth")
    with pytest.raises(RuntimeError, match="extra"):
        PretrainedSTGCN(16, tmp_path / "bad.pth")


def test_random_start_when_no_checkpoint() -> None:
    torch.manual_seed(0)
    a = PretrainedSTGCN(5)
    torch.manual_seed(1)
    b = PretrainedSTGCN(5)
    assert not torch.equal(a.gcn[0].gcn.conv.weight, b.gcn[0].gcn.conv.weight)
    assert torch.equal(a.gcn[0].gcn.A, coco_spatial_graph())
    assert torch.equal(a.gcn[0].gcn.PA, torch.ones(3, 17, 17))


def test_export_form_gives_the_same_logits(tmp_path: Path) -> None:
    _fake_checkpoint(tmp_path / "ckpt.pth")
    model = PretrainedSTGCN(16, tmp_path / "ckpt.pth").eval()
    with torch.no_grad():  # trained importance weights, so the fold must carry A * PA
        for block in model.gcn:
            block.gcn.PA.uniform_(0.5, 1.5)
    x = torch.rand(4, 3, 30, 33)
    x[:, 2] = (x[:, 2] > 0.2).float() * x[:, 2]
    x[:, :2] *= (x[:, 2:] > 0).float()  # missing points are (0, 0, 0), as the build writes them
    mobile = export_form(model)
    with torch.no_grad():
        assert torch.allclose(mobile(coco17_input(x)), model(x), atol=1e-4)


def test_export_form_has_no_float_island_layers() -> None:
    mobile = export_form(PretrainedSTGCN(5).eval())
    # D-53 check 4 runs 3-5: no BatchNorm after a graph step, no einsum, no subset-sum ADD
    for block in mobile.gcn:
        assert not any(isinstance(m, nn.BatchNorm1d | nn.BatchNorm2d) for m in block.gcn.modules())
        assert block.gcn.mix.in_features == 3 * 17 and block.gcn.mix.out_features == 17
    assert not any(isinstance(m, nn.BatchNorm1d) for m in mobile.modules())
    torch.export.export(mobile, (torch.zeros(2, 3, 30, 17),))
