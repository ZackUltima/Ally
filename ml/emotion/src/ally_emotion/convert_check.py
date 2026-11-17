"""Can MobileNetV3 Large become an INT8 phone model? (EM1a Task 5; technical plan E12, D-53, D-57)

D-53 rule: a model ends with a fixed-size average pool, because an adaptive pool stays in float
after conversion. MobileNetV3 has one at the end and one inside every squeeze-and-excitation
block, so `fix_pools` swaps each for a fixed `AvgPool2d` of the size measured at 224 x 224.

The check follows the D-53 notebook (`research/spikes/d53/`): the model with fixed pools is
prepared for quantisation-aware training (QAT) and, separately, post-training quantised (PTQ)
with 200 FER+ training crops, converted with LiteRT Torch, and the operations are listed with the
default delegate off. Pass: it converts and runs, and every compute operation works on INT8
tensors. Reported without a mark: float against INT8 top-1 agreement on FER+ validation crops.

Needs torch, torchvision and the converter (the `ally-emotion` environment); CI has none of them,
so this module is left out of the coverage count and its tests skip themselves there.

Usage: python -m ally_emotion.convert_check --data-root ~/ally-data
       [--arch mobilenet_v2] [--pretrained]
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import logging
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.nn import functional

logger = logging.getLogger(__name__)

INPUT_SIZE = 224
CALIBRATION_CROPS = 200
AGREEMENT_CROPS = 500
CLASSES = 8  # the FER+ classes
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
COMPUTE_OPS = frozenset({
    "CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "AVERAGE_POOL_2D", "ADD", "MUL",
    "MEAN", "SUM", "HARD_SWISH", "LOGISTIC", "RELU", "BATCH_MATMUL"})


# ------------------------------------------------------------------------------ the model

def fix_pools(model: nn.Module, input_size: int = INPUT_SIZE) -> nn.Module:
    """A copy of `model` in which every `AdaptiveAvgPool2d` is a fixed `AvgPool2d` whose window
    is the feature-map size that pool sees at `input_size` (the original is not changed)."""
    fixed = copy.deepcopy(model).eval()
    seen: dict[str, tuple[int, int]] = {}
    hooks = [
        module.register_forward_hook(
            lambda _m, inputs, _out, key=name: seen.__setitem__(key, tuple(inputs[0].shape[-2:])))
        for name, module in fixed.named_modules() if isinstance(module, nn.AdaptiveAvgPool2d)]
    with torch.no_grad():
        fixed(torch.zeros(1, 3, input_size, input_size))
    for hook in hooks:
        hook.remove()
    for name, size in seen.items():
        parent_name, _, child = name.rpartition(".")
        parent = fixed.get_submodule(parent_name) if parent_name else fixed
        setattr(parent, child, nn.AvgPool2d(kernel_size=size))
    return fixed


class _FixedMobileNetV2(nn.Module):
    """torchvision's MobileNetV2 calls the adaptive pool as a function inside `forward`, which
    `fix_pools` cannot reach; this wrapper has the same layers with a fixed 7 x 7 pool."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.features = base.features
        self.pool = nn.AvgPool2d(kernel_size=INPUT_SIZE // 32)
        self.classifier = base.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.flatten(self.pool(self.features(x)), 1))


ARCHITECTURES = ("mobilenet_v3_large", "mobilenet_v2")


def build_model(arch: str, pretrained: bool, classes: int = CLASSES) -> nn.Module:
    """The architecture with fixed-size pools and a `classes`-way head (default: the 8 FER+
    classes), in eval mode. `pretrained` loads the ImageNet weights (downloaded once by
    torchvision)."""
    import torchvision

    if arch == "mobilenet_v3_large":
        weights = (torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
                   if pretrained else None)
        model = torchvision.models.mobilenet_v3_large(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, classes)
        return fix_pools(model)
    if arch == "mobilenet_v2":
        weights = (torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
                   if pretrained else None)
        model = torchvision.models.mobilenet_v2(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, classes)
        return _FixedMobileNetV2(model).eval()
    raise ValueError(f"unknown architecture {arch!r}; use one of {ARCHITECTURES}")


def prepare_crops(crops: NDArray[np.uint8]) -> NDArray[np.float32]:
    """FER+ crops (n, 48, 48) grey -> (n, 3, 224, 224) float32: bilinear resize, the grey copied
    to three channels, ImageNet mean and standard deviation. For this check only; the phone
    path is the Kotlin-portable crop in `crop.py`."""
    x = torch.from_numpy(crops.astype(np.float32) / 255.0)[:, None]
    x = functional.interpolate(x, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear",
                               align_corners=False)
    out = x.repeat(1, 3, 1, 1).numpy()
    return ((out - _MEAN[None, :, None, None]) / _STD[None, :, None, None]).astype(np.float32)


# ------------------------------------------------------------------------------ conversion

def _pt2e_helper(name: str) -> Callable[[nn.Module], None]:
    for module_name in ("torchao.quantization.pt2e", "torchao.quantization.pt2e.utils",
                        "torch.ao.quantization"):
        try:
            fn = getattr(importlib.import_module(module_name), name, None)
        except ImportError:
            fn = None
        if fn is not None:
            return fn
    raise ImportError(f"{name} not found in torchao or torch.ao")


class _PatternModule(nn.Module):
    def __init__(self, fn: Callable[..., object]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, *args: object, **kwargs: object) -> object:
        return self.fn(*args, **kwargs)


def _patch_pattern_export() -> None:
    """Same compatibility fix as the D-53 spike and `ally_fall.export`: QAT's conv + batch-norm
    patterns are plain functions, which current torchao cannot export."""
    import litert_torch.quantize.pt2e_quantizer_utils as utils

    original = utils._get_aten_graph_module_for_pattern
    if getattr(original, "_ally_patched", False):
        return

    def wrapped(pattern: object, *args: object, **kwargs: object) -> object:
        if not isinstance(pattern, nn.Module):
            pattern = _PatternModule(pattern)
        return original(pattern, *args, **kwargs)

    wrapped._ally_patched = True  # type: ignore[attr-defined]
    utils._get_aten_graph_module_for_pattern = wrapped


def export_graph(model: nn.Module, example: NDArray[np.float32], train: bool) -> nn.Module:
    """The model exported with a flexible batch size, as a graph the quantiser can prepare."""
    batch = torch.export.Dim("batch", min=1, max=256)
    return torch.export.export(copy.deepcopy(model).train(train), (torch.from_numpy(example),),
                               dynamic_shapes=({0: batch},)).module()


def tag_adds(graph: nn.Module) -> int:
    """Give every `aten.add` node the source mark the quantiser looks for and returns how many
    it marked. Found in this check (2026-10-29): after `torch.export` the add nodes carry no
    `source_fn_stack`, so LiteRT Torch's quantiser finds no add and leaves every residual add in
    float32."""
    marked = 0
    for node in graph.graph.nodes:
        if node.op == "call_function" and node.target is torch.ops.aten.add.Tensor:
            node.meta["source_fn_stack"] = [(node.name, torch.add)]
            marked += 1
    return marked


def convert_int8(model: nn.Module, method: str, calibration: NDArray[np.float32],
                 path: Path) -> None:
    """Write an INT8 `.tflite` of `model` to `path`. `qat` = prepared for QAT (INT8 rounding
    simulated, batch-norm folded) and passed over the calibration crops once, with no weight
    training; `ptq` = prepared and calibrated on the same crops."""
    if method not in ("qat", "ptq"):
        raise ValueError(f"method must be 'qat' or 'ptq', not {method!r}")
    _patch_pattern_export()
    import litert_torch
    from litert_torch.quantize.pt2e_quantizer import (
        PT2EQuantizer,
        get_symmetric_quantization_config,
    )
    from litert_torch.quantize.quant_config import QuantConfig
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e, prepare_qat_pt2e

    qat = method == "qat"
    example = torch.from_numpy(calibration[:2])
    quantizer = PT2EQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_qat=qat))
    graph = export_graph(model, calibration[:2], train=qat)
    logger.info("%s: %d residual adds tagged for the quantiser", method, tag_adds(graph))
    graph = (prepare_qat_pt2e if qat else prepare_pt2e)(graph, quantizer)
    if qat:
        _pt2e_helper("move_exported_model_to_train")(graph)
    with torch.no_grad():
        for start in range(0, len(calibration), 16):
            graph(torch.from_numpy(calibration[start:start + 16]))
    if qat:
        _pt2e_helper("move_exported_model_to_eval")(graph)
    quantised = convert_pt2e(graph, fold_quantize=False)
    edge = litert_torch.convert(quantised, (example[:1],),
                                quant_config=QuantConfig(pt2e_quantizer=quantizer))
    path.parent.mkdir(parents=True, exist_ok=True)
    edge.export(str(path))


def mean_cosine(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Mean cosine similarity between matching rows of two score matrices (0 for a row with no
    direction). A steadier agreement figure than top-1 when the head is untrained and its scores
    are nearly tied."""
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    dots = (a * b).sum(axis=1)
    return float(np.where(norms > 0, dots / np.where(norms > 0, norms, 1.0), 0.0).mean())


@dataclass(frozen=True)
class TfliteRun:
    size_mb: float
    ms_per_crop: float
    ops: dict[str, list[str]]
    float_ops: tuple[str, ...]
    preds: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    logits: NDArray[np.float32] = field(default_factory=lambda: np.zeros((0, 0), np.float32))


def run_tflite(path: Path, crops: NDArray[np.float32]) -> TfliteRun:
    """Run `path` on `crops` one at a time in the LiteRT interpreter and list the operations
    that still compute in float32, with the default delegate off (D-53 run 2 lesson)."""
    from ai_edge_litert.interpreter import Interpreter, OpResolverType

    it = Interpreter(model_path=str(path))
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    preds: list[int] = []
    scores: list[NDArray[np.float32]] = []
    times: list[float] = []
    for crop in crops:
        x = crop[None]
        if inp["dtype"] == np.int8:
            scale, zero = inp["quantization"]
            x = np.clip(np.round(x / scale + zero), -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], x)
        started = time.perf_counter()
        it.invoke()
        times.append(time.perf_counter() - started)
        raw = it.get_tensor(out["index"])[0].astype(np.float32)
        if out["dtype"] == np.int8:
            scale, zero = out["quantization"]
            raw = (raw - zero) * scale
        scores.append(raw)
        preds.append(int(np.argmax(raw)))
    plain = Interpreter(
        model_path=str(path),
        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    plain.allocate_tensors()
    dtype_of = {d["index"]: np.dtype(d["dtype"]).name for d in plain.get_tensor_details()}
    ops: dict[str, set[str]] = {}
    for op in plain._get_ops_details():
        first = [i for i in op["inputs"] if i >= 0][:1]
        ops.setdefault(op["op_name"], set()).add(dtype_of.get(first[0], "?") if first else "-")
    return TfliteRun(
        size_mb=path.stat().st_size / 2**20,
        ms_per_crop=1000 * float(np.median(times)) if times else 0.0,
        ops={k: sorted(v) for k, v in sorted(ops.items())},
        float_ops=tuple(sorted(k for k, v in ops.items() if k in COMPUTE_OPS and "float32" in v)),
        preds=np.array(preds, dtype=np.int64), logits=np.stack(scores))


# --------------------------------------------------------------------------------- the check

@dataclass(frozen=True)
class CheckResult:
    method: str
    converts: bool
    all_int8: bool
    float_ops: list[str]
    top1_agreement: float
    logit_cosine: float
    size_mb: float
    ms_per_crop: float
    error: str = ""


def run_check(out: Path, data_root: Path, arch: str = "mobilenet_v3_large",
              pretrained: bool = False) -> list[CheckResult]:
    """Convert `arch` with fixed pools both ways and report. FER+ training crops
    calibrate; FER+ validation crops measure float against INT8 agreement."""
    built = data_root / "built" / "emotion"
    with np.load(built / "ferplus_train.npz") as f:
        train = f["images"]
    with np.load(built / "ferplus_val.npz") as f:
        val = f["images"]
    rng = np.random.default_rng(0)
    calibration = prepare_crops(train[rng.choice(len(train), CALIBRATION_CROPS, replace=False)])
    held = prepare_crops(val[rng.choice(len(val), AGREEMENT_CROPS, replace=False)])
    torch.manual_seed(0)
    model = build_model(arch, pretrained)
    with torch.no_grad():
        float_logits = np.concatenate([
            model(torch.from_numpy(held[i:i + 50])).numpy() for i in range(0, len(held), 50)])
    float_preds = float_logits.argmax(1)
    results = []
    for method in ("ptq", "qat"):
        path = out / f"{arch}_{method}.tflite"
        try:
            convert_int8(model, method, calibration, path)
            run = run_tflite(path, held)
            results.append(CheckResult(
                method, True, not run.float_ops, list(run.float_ops),
                float((run.preds == float_preds).mean()), mean_cosine(run.logits, float_logits),
                run.size_mb, run.ms_per_crop))
            (out / f"{arch}_{method}_ops.json").write_text(
                json.dumps(run.ops, indent=2) + "\n", encoding="utf-8")
        except Exception as error:  # a failed conversion is the result being reported
            logger.exception("%s conversion failed", method)
            results.append(CheckResult(method, False, False, [], 0.0, 0.0, 0.0, 0.0, repr(error)))
    return results


def report_markdown(results: Sequence[CheckResult], arch: str, pretrained: bool) -> str:
    start = "ImageNet-pretrained" if pretrained else "random weights (structure check only)"
    lines = [
        f"# {arch} INT8 conversion check (EM1a Task 5, D-53)", "",
        f"Model: torchvision {arch}, 8-class head, fixed pools, start: {start}. "
        f"Calibration: {CALIBRATION_CROPS} FER+ training crops; agreement on "
        f"{AGREEMENT_CROPS} FER+ validation crops (48 x 48 grey, resized to 224 x 224, "
        "three channels). Converter: "
        "LiteRT Torch; operations listed with the default delegate off. **Pass** = converts, runs "
        "and every compute operation works on INT8 tensors. Agreement is reported with no mark: "
        "the 98% mark of D-53 applies to the trained model in EM2. The 8-class head is new and "
        "untrained, so its scores are nearly tied and top-1 flips easily; the mean cosine of the "
        "float and INT8 scores is the steadier figure.", "",
        "| method | converts | all compute ops INT8 | float32 compute ops | float vs INT8 top-1 "
        "| scores, mean cosine | size | ms per crop (laptop CPU) |",
        "|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(
            f"| {r.method.upper() if r.method == 'ptq' else 'QAT-prepared'} | "
            f"{'yes' if r.converts else 'NO'} | {'yes' if r.all_int8 else 'no'} | "
            f"{', '.join(r.float_ops) or 'none'} | {100 * r.top1_agreement:.1f}% | "
            f"{r.logit_cosine:.4f} | "
            f"{r.size_mb:.1f} MB | {r.ms_per_crop:.1f} |")
    for r in results:
        if r.error:
            lines += ["", f"{r.method}: {r.error}"]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--arch", choices=ARCHITECTURES, default="mobilenet_v3_large")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args(argv)
    root = args.data_root.expanduser()
    out = args.out or root / "checks" / args.arch
    results = run_check(out, root, args.arch, args.pretrained)
    name = args.arch.replace("_", "")
    report = Path(__file__).resolve().parents[2] / "results" / "checks" / f"{name}-int8.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    text = report_markdown(results, args.arch, args.pretrained)
    report.write_text(text, encoding="utf-8", newline="\n")
    report.with_suffix(".json").write_text(
        json.dumps([asdict(r) for r in results], indent=2) + "\n", encoding="utf-8")
    logger.info("report written to %s", report)
    return 0 if all(r.converts and r.all_int8 for r in results) else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(main())
