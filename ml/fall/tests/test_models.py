import pytest
import torch
from torch import nn

from ally_fall.models import GRUNet, ModelName, build_model

NAMES: tuple[ModelName, ...] = ("tcn", "gru", "stgcn")


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("batch", [1, 7])
@pytest.mark.parametrize("n_classes", [16, 5])
def test_output_shape(name: ModelName, batch: int, n_classes: int) -> None:
    model = build_model(name, n_classes, hidden=16).eval()
    assert model(torch.zeros(batch, 3, 30, 33)).shape == (batch, n_classes)


@pytest.mark.parametrize("name", ["tcn", "stgcn"])
def test_no_adaptive_pool(name: ModelName) -> None:
    model = build_model(name, 5, hidden=16)
    assert not any("Adaptive" in type(m).__name__ for m in model.modules())


def test_gru_uses_explicit_start_state_and_both_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    original = nn.GRU.forward

    def recording(self: nn.GRU, *args: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(len(args))
        return original(self, *args)

    monkeypatch.setattr(nn.GRU, "forward", recording)
    model = GRUNet(5, hidden=8).eval()
    model(torch.zeros(2, 3, 30, 33))
    assert calls == [2]  # the input and an explicit start state
    with torch.no_grad():  # the logits use both outputs: change h_n alone and they move
        x = torch.randn(1, 3, 30, 33)
        out, h_n = model.gru(x.permute(0, 2, 1, 3).reshape(1, 30, 99), torch.zeros(1, 1, 8))
        assert torch.allclose(model.head(out, h_n), model.fc((out[:, -1] + h_n[-1]) * 0.5))


def test_build_model_pretrained_candidate() -> None:
    model = build_model("stgcn_pt", 5, hidden=16).eval()  # hidden is fixed by the checkpoint
    assert type(model).__name__ == "PretrainedSTGCN"
    assert model(torch.zeros(2, 3, 30, 33)).shape == (2, 5)


def test_build_model_unknown_name() -> None:
    with pytest.raises(ValueError, match="lstm"):
        build_model("lstm", 5)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", NAMES)
def test_exportable(name: ModelName) -> None:
    model = build_model(name, 5, hidden=16).eval()
    torch.export.export(model, (torch.zeros(2, 3, 30, 33),))


@pytest.mark.parametrize("name", NAMES)
def test_overfits_tiny_batch(name: ModelName) -> None:
    torch.manual_seed(0)
    x = torch.randn(32, 3, 30, 33)
    y = (x[:, 1, -1, 23] > 0).long()  # class = whether the last hip y is above 0
    x[:, 1, -1, 23] += torch.where(y == 1, 2.0, -2.0)
    model = build_model(name, 2, hidden=16)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(200):
        optimiser.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        optimiser.step()
    model.eval()
    assert (model(x).argmax(1) == y).float().mean().item() == 1.0
