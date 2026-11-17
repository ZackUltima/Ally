"""Task 5: the MobileNetV3 pooling fix and the crop preparation for the INT8 check (D-53 rule:
models end with a fixed-size average pool). Skipped where torch or torchvision is missing: CI
installs CPU torch without torchvision (converter pins stay out of CI until F3)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

from ally_emotion.convert_check import (  # noqa: E402
    build_model,
    export_graph,
    fix_pools,
    mean_cosine,
    prepare_crops,
    tag_adds,
)


def mnv3() -> torch.nn.Module:
    torch.manual_seed(0)
    return torchvision.models.mobilenet_v3_large(weights=None).eval()


def test_no_adaptive_pool_is_left() -> None:
    fixed = fix_pools(mnv3())
    assert not any(isinstance(m, torch.nn.AdaptiveAvgPool2d) for m in fixed.modules())


def test_outputs_are_unchanged_at_224() -> None:
    original = mnv3()
    fixed = fix_pools(mnv3())  # same seed, same weights
    for seed in range(4):
        x = torch.from_numpy(np.random.default_rng(seed).normal(size=(2, 3, 224, 224))
                             .astype(np.float32))
        with torch.no_grad():
            assert (original(x) - fixed(x)).abs().max().item() < 1e-5


def test_the_se_blocks_no_longer_call_adaptive_pooling() -> None:
    graph = torch.fx.symbolic_trace(fix_pools(mnv3()))
    modules = dict(graph.named_modules())
    for node in graph.graph.nodes:
        if node.op == "call_module":
            assert not isinstance(modules[node.target], torch.nn.AdaptiveAvgPool2d)
        if node.op == "call_function":
            assert "adaptive_avg_pool2d" not in getattr(node.target, "__name__", "")


def test_fix_pools_does_not_change_the_model_it_is_given() -> None:
    model = mnv3()
    fix_pools(model)
    assert any(isinstance(m, torch.nn.AdaptiveAvgPool2d) for m in model.modules())


def test_prepare_crops_gives_normalised_three_channel_224_input() -> None:
    crops = np.full((2, 48, 48), 128, dtype=np.uint8)
    out = prepare_crops(crops)
    assert out.shape == (2, 3, 224, 224) and out.dtype == np.float32
    # a flat grey crop is the same in all three channels before the ImageNet mean and std
    expected = (128 / 255 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    assert np.allclose(out[0, :, 0, 0], expected, atol=1e-5)


def test_mobilenet_v2_ends_with_a_fixed_pool_and_matches_torchvision() -> None:
    torch.manual_seed(0)
    original = torchvision.models.mobilenet_v2(weights=None).eval()
    torch.manual_seed(0)
    fixed = build_model("mobilenet_v2", pretrained=False, classes=1000)
    fixed.classifier.load_state_dict(original.classifier.state_dict())  # the head is new by design
    assert not any(isinstance(m, torch.nn.AdaptiveAvgPool2d) for m in fixed.modules())
    x = torch.from_numpy(np.random.default_rng(0).normal(size=(2, 3, 224, 224)).astype(np.float32))
    with torch.no_grad():
        assert (original(x) - fixed(x)).abs().max().item() < 1e-5


def test_an_unknown_architecture_is_refused() -> None:
    with pytest.raises(ValueError, match="mobilenet_v4"):
        build_model("mobilenet_v4", pretrained=False)


def test_the_quantiser_finds_every_residual_add_once_they_are_tagged() -> None:
    import operator

    from torch.fx.passes.utils.source_matcher_utils import get_source_partitions

    model = build_model("mobilenet_v2", pretrained=False)
    graph = export_graph(model, np.zeros((2, 3, 224, 224), dtype=np.float32), train=False)
    adds = [n for n in graph.graph.nodes if n.op == "call_function" and "add" in str(n.target)]

    def found() -> int:
        parts = get_source_partitions(graph.graph, [operator.add, torch.add, operator.iadd])
        return sum(len(v) for v in parts.values())

    assert len(adds) == 10 and found() == 0  # MobileNetV2 has 10 residual adds, none visible
    assert tag_adds(graph) == 10
    assert found() == 10


def test_mean_cosine_is_one_for_equal_and_minus_one_for_opposite_scores() -> None:
    a = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 0.0]], dtype=np.float32)
    assert mean_cosine(a, a) == pytest.approx(1.0)
    assert mean_cosine(a, -a) == pytest.approx(-1.0)
    assert mean_cosine(a, np.zeros_like(a)) == 0.0  # no direction: reported as 0, not NaN
