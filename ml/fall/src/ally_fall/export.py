"""INT8 export of a trained fall model, with the D-53 marks and the D-68 gate (F3b Task 5).

A trained PyTorch run becomes a `.tflite` file whose calculations run on INT8 numbers: first
by quantisation-aware training (QAT, a short extra fit with INT8 rounding simulated), or by
post-training quantisation (PTQ, calibrated on training windows only) as the fallback (D-20).
The file is run on the validation windows in the LiteRT interpreter and compared with the
float model. Validation only: the test split is not opened here (it is `final_test.py`'s, once).

Usage (needs LiteRT Torch, pinned in requirements.txt; Linux only):
  python -m ally_fall.export --config configs/f2-grid/<run>.toml [--config ...] \\
      --data-root ~/ally-data --out results/f3b/int8.md
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import logging
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from ally_fall.augment import augment_batch
from ally_fall.data import load_split, window_tensor
from ally_fall.labels import class_names, fall_classes, label_map
from ally_fall.models import build_model
from ally_fall.omnifall import label_names
from ally_fall.train import RunConfig, load_config, model_input, score
from ally_fall.weights import class_weights

logger = logging.getLogger(__name__)

Method = Literal["qat", "ptq"]
Batches = Callable[[], Iterable[tuple[NDArray[np.float32], NDArray[np.int64]]]]

# The D-53 marks, fixed before any run (D-68 decision 3), and the gate on the fall decision.
MAX_ACCURACY_DROP = 0.02  # INT8 label accuracy at most 2 points below float
MIN_AGREEMENT = 0.98  # same answer as float on at least 98% of windows
MAX_FEWER_FALLS = 2  # the gate: the INT8 file may find at most one fewer fall than float
QAT_LR = 1e-4  # the D-53 spike's rate for the short extra fit
CALIBRATION_WINDOWS = 256  # PTQ calibration size, as in the spike
_EPS = 1e-9

# Operations that do arithmetic; a float input to one of these is a float island (D-53 mark 1).
COMPUTE_OPS = frozenset({
    "CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "BATCH_MATMUL", "ADD", "SUB", "MUL",
    "MEAN", "SUM", "AVERAGE_POOL_2D", "LOGISTIC", "TANH", "UNIDIRECTIONAL_SEQUENCE_RNN",
    "UNIDIRECTIONAL_SEQUENCE_LSTM",
})


@dataclass(frozen=True)
class Marks:
    """The INT8 file against the float model on the same validation windows."""

    accuracy_float: float
    accuracy_int8: float
    label_agreement: float  # same top label (all 16 classes)
    fall_agreement: float  # same fall / not-fall decision (the product acts on this)
    falls_found_float: int
    falls_found_int8: int
    float_compute_ops: tuple[str, ...]

    @property
    def accuracy_drop(self) -> float:
        return self.accuracy_float - self.accuracy_int8

    @property
    def ops_pass(self) -> bool:
        return not self.float_compute_ops

    @property
    def accuracy_pass(self) -> bool:
        return self.accuracy_drop <= MAX_ACCURACY_DROP + _EPS

    @property
    def agreement_pass(self) -> bool:
        return self.label_agreement >= MIN_AGREEMENT - _EPS

    @property
    def gate_pass(self) -> bool:
        """The D-68 gate: fall decision agrees on >= 98% of windows, <= 1 fall lost."""
        return (self.fall_agreement >= MIN_AGREEMENT - _EPS
                and self.falls_found_float - self.falls_found_int8 < MAX_FEWER_FALLS)


def compute_marks(
    float_pred: NDArray[np.int64],
    int8_pred: NDArray[np.int64],
    y: NDArray[np.int64],
    fall: tuple[int, ...],
    falls_found_float: int,
    falls_found_int8: int,
    float_compute_ops: Sequence[str],
) -> Marks:
    if not (len(float_pred) == len(int8_pred) == len(y)):
        raise ValueError("float answers, INT8 answers and labels must be for the same windows")
    if len(y) == 0:
        raise ValueError("no windows to compare")
    fall_float, fall_int8 = np.isin(float_pred, fall), np.isin(int8_pred, fall)
    return Marks(
        accuracy_float=float((float_pred == y).mean()),
        accuracy_int8=float((int8_pred == y).mean()),
        label_agreement=float((float_pred == int8_pred).mean()),
        fall_agreement=float((fall_float == fall_int8).mean()),
        falls_found_float=falls_found_float,
        falls_found_int8=falls_found_int8,
        float_compute_ops=tuple(float_compute_ops),
    )


@dataclass(frozen=True)
class TfliteRun:
    """A `.tflite` file run on windows: its answers, size, speed and operations."""

    size_kb: float
    ms_per_window: float
    ops: dict[str, list[str]]
    float_compute_ops: tuple[str, ...]
    preds: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


def epoch_batches(n: int, batch: int, rng: np.random.Generator) -> Iterator[NDArray[np.int64]]:
    """The window indices of one epoch in random order; a last batch of one is dropped (batch
    norm needs two windows, as in `train.train`)."""
    order = rng.permutation(n).astype(np.int64)
    for start in range(0, n, batch):
        chunk = order[start:start + batch]
        if len(chunk) >= 2:
            yield chunk


# ----------------------------------------------------------------------------- conversion

def _pt2e_helper(name: str) -> Callable[[nn.Module], None]:
    """`move_exported_model_to_train/_eval` live in torchao or in torch.ao, by version."""
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
    """LiteRT Torch passes plain functions as conv + batch-norm patterns, but current torchao
    exports them with `torch.export`, which only takes an `nn.Module`; wrapping each function
    restores the QAT path (found in the D-53 spike, 2026-10-17)."""
    import litert_torch.quantize.pt2e_quantizer_utils as utils

    original = utils._get_aten_graph_module_for_pattern
    if getattr(original, "_ally_patched", False):
        return

    def wrapped(pattern: object, *args: object, **kwargs: object) -> object:
        if not isinstance(pattern, nn.Module):
            pattern = _PatternModule(pattern)
        return original(pattern, *args, **kwargs)

    wrapped._ally_patched = True
    utils._get_aten_graph_module_for_pattern = wrapped


def _exported(model: nn.Module, example: NDArray[np.float32]) -> nn.Module:
    """Export with a flexible batch size, so the graph trains on batches and later runs one
    window at a time."""
    batch = torch.export.Dim("batch", min=1, max=4096)
    return torch.export.export(model, (torch.from_numpy(example),),
                               dynamic_shapes=({0: batch},)).module()


def convert_int8(
    model: nn.Module,
    method: str,
    batches: Batches,
    weights: NDArray[np.float32],
    path: Path,
    epochs: int,
    lr: float = QAT_LR,
) -> None:
    """Write an INT8 `.tflite` of `model` (float, trained) to `path`.

    QAT: `epochs` passes over `batches()` with INT8 rounding simulated, class-weighted cross
    entropy as in training. PTQ: calibrated on the first `CALIBRATION_WINDOWS` windows of
    `batches()`; `epochs` is ignored. Both use training windows only.
    """
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

    first_x, _ = next(iter(batches()))
    example = first_x[:2]
    qat = method == "qat"
    quantizer = PT2EQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_qat=qat))
    graph = _exported(copy.deepcopy(model).train(qat), example)
    to_train, to_eval = (_pt2e_helper("move_exported_model_to_train"),
                         _pt2e_helper("move_exported_model_to_eval"))
    if qat:
        graph = prepare_qat_pt2e(graph, quantizer)
        optimiser = torch.optim.Adam(graph.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights))
        for epoch in range(epochs):
            to_train(graph)
            total, seen = 0.0, 0
            for xb, yb in batches():
                optimiser.zero_grad()
                loss = loss_fn(graph(torch.from_numpy(xb)), torch.from_numpy(yb))
                loss.backward()
                optimiser.step()
                total += loss.item() * len(xb)
                seen += len(xb)
            logger.info("QAT epoch %d/%d: loss %.4f", epoch + 1, epochs, total / max(seen, 1))
        to_eval(graph)
    else:
        graph = prepare_pt2e(graph, quantizer)
        left = CALIBRATION_WINDOWS
        with torch.no_grad():
            for xb, _ in batches():
                graph(torch.from_numpy(xb[:left]))
                left -= min(left, len(xb))
                if left <= 0:
                    break
    quantised = convert_pt2e(graph, fold_quantize=False)
    edge = litert_torch.convert(quantised, (torch.from_numpy(example[:1]),),
                                quant_config=QuantConfig(pt2e_quantizer=quantizer))
    path.parent.mkdir(parents=True, exist_ok=True)
    edge.export(str(path))


# --------------------------------------------------------------------------- running a file

def run_tflite(path: Path, windows: NDArray[np.float32]) -> TfliteRun:
    """Run `path` on `windows` (n, 3, ticks, 33), one at a time, in the LiteRT interpreter.

    `float_compute_ops` lists arithmetic operations that still compute in float32 ("float
    islands"); the list is made with the default delegate off, because a delegated operation
    hides inside one node whose number type cannot be seen (D-53 spike, run 1).
    """
    from ai_edge_litert.interpreter import Interpreter, OpResolverType

    it = Interpreter(model_path=str(path))
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    preds: list[int] = []
    times: list[float] = []
    for window in windows:
        x = window[None]
        if inp["dtype"] == np.int8:  # the file wants INT8 input
            scale, zero = inp["quantization"]
            x = np.clip(np.round(x / scale + zero), -128, 127).astype(np.int8)
        it.set_tensor(inp["index"], x)
        started = time.perf_counter()
        it.invoke()
        times.append(time.perf_counter() - started)
        preds.append(int(np.argmax(it.get_tensor(out["index"]))))
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
        size_kb=path.stat().st_size / 1024,
        ms_per_window=1000 * float(np.median(times)) if times else 0.0,
        ops={k: sorted(v) for k, v in sorted(ops.items())},
        float_compute_ops=tuple(sorted(k for k, v in ops.items()
                                       if k in COMPUTE_OPS and "float32" in v)),
        preds=np.array(preds, dtype=np.int64),
    )


# ------------------------------------------------------------------------------- reporting

def _verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def report_markdown(run_id: str, method: str, m: Marks, run: TfliteRun, falls_total: int) -> str:
    ops = ", ".join(f"{name} ({'/'.join(kinds)})" for name, kinds in run.ops.items())
    float_ops = ", ".join(m.float_compute_ops) or "none"
    return "\n".join([
        f"## {run_id}: INT8 by {method.upper()}",
        "",
        "| D-53 mark (fixed before any run) | Result | |",
        "|---|---|---|",
        f"| every calculation on INT8 | float calculations: {float_ops} | {_verdict(m.ops_pass)} |",
        f"| accuracy at most 2 points below float | float {m.accuracy_float:.4f}, INT8 "
        f"{m.accuracy_int8:.4f}, drop {m.accuracy_drop:+.4f} | {_verdict(m.accuracy_pass)} |",
        f"| same label as float on at least 98% of windows | {m.label_agreement:.4f} "
        f"(all 16 labels) | {_verdict(m.agreement_pass)} |",
        "",
        "| Gate on the fall decision (D-68 decision 3) | Result | |",
        "|---|---|---|",
        f"| fall / not-fall decision agrees with float, at least 98% | "
        f"{m.fall_agreement:.4f} | {_verdict(m.fall_agreement >= MIN_AGREEMENT - _EPS)} |",
        f"| falls found: INT8 loses at most 1 of the float model's | float "
        f"{m.falls_found_float} of {falls_total}, INT8 {m.falls_found_int8} of {falls_total} | "
        f"{_verdict(m.falls_found_float - m.falls_found_int8 < MAX_FEWER_FALLS)} |",
        f"| **gate** | | **{_verdict(m.gate_pass)}** |",
        "",
        f"File: {run.size_kb:.1f} KB. Laptop CPU: {run.ms_per_window:.3f} ms per window "
        "(median, one window at a time). Phone speed and heat are not measured here (Spike 2, "
        "D-40).",
        "",
        f"Operations (input type): {ops}.",
        "",
    ])


# ---------------------------------------------------------------------------- the whole run

@dataclass(frozen=True)
class ExportResult:
    run_id: str
    method: str
    marks: Marks
    run: TfliteRun
    falls_total: int
    path: Path


def export_run(cfg: RunConfig, data_root: Path, weights: Path, out_dir: Path, method: Method,
               epochs: int, batch: int = 256) -> ExportResult:
    """Convert one trained run to INT8 and score both models on the validation people."""
    names = label_names(data_root / "omnifall")
    classes = class_names(cfg.label_set, names)
    fall = fall_classes(cfg.label_set, names)
    mapping = label_map(names, cfg.label_set)
    built = data_root / "built" / cfg.pose_model
    train_split = load_split(built / "train.npz", mapping)
    val = load_split(built / "val.npz", mapping)

    model = build_model(cfg.model, len(classes), cfg.hidden, None, cfg.window_ticks)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval()
    rng = np.random.default_rng(cfg.seed)

    def source() -> Iterator[tuple[NDArray[np.float32], NDArray[np.int64]]]:
        for idx in epoch_batches(len(train_split.window_end), batch, rng):
            windows = augment_batch(window_tensor(train_split, idx, cfg.window_ticks), rng,
                                    cfg.augment)
            yield model_input(windows, cfg), train_split.y[idx]

    counts = np.bincount(train_split.y, minlength=len(classes))
    path = out_dir / f"{cfg.run_id}-int8-{method}.tflite"
    convert_int8(model, method, source, class_weights(counts, cfg.beta).astype(np.float32),
                 path, epochs)

    index = np.arange(len(val.window_end), dtype=np.int64)
    x = model_input(window_tensor(val, index, cfg.window_ticks), cfg)
    with torch.no_grad():
        float_pred = model(torch.from_numpy(x)).argmax(1).numpy().astype(np.int64)
    run = run_tflite(path, x)
    found = {
        label: score(pred, np.isin(pred, fall), val, fall, classes).events
        for label, pred in (("float", float_pred), ("int8", run.preds))
    }
    marks = compute_marks(float_pred, run.preds, val.y, fall, found["float"].detected,
                          found["int8"].detected, run.float_compute_ops)
    total = found["float"].detected + found["float"].missed
    return ExportResult(cfg.run_id, method, marks, run, total, path)


def results_markdown(results: Sequence[ExportResult]) -> str:
    head = [
        "# F3b INT8 export (validation people only; the test split is untouched)",
        "",
        "Each trained run is converted to an INT8 `.tflite` and run on the same validation "
        "windows as the float model. The D-53 marks are reported as fixed; the gate is on the "
        "fall / not-fall decision (D-68 decision 3). The comparison is on windows, so a window "
        "count is not an event count: 'falls found' is the event count, as in the model choice.",
        "",
        "| Run | Method | Label agreement | Fall agreement | Falls found (float / INT8) | KB | "
        "D-53 marks | Gate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        m = r.marks
        marks = _verdict(m.ops_pass and m.accuracy_pass and m.agreement_pass)
        found = f"{m.falls_found_float} / {m.falls_found_int8} of {r.falls_total}"
        head.append(f"| {r.run_id} | {r.method.upper()} | {m.label_agreement:.4f} | "
                    f"{m.fall_agreement:.4f} | {found} | {r.run.size_kb:.1f} | {marks} | "
                    f"{_verdict(m.gate_pass)} |")
    head.append("")
    return "\n".join(head + [report_markdown(r.run_id, r.method, r.marks, r.run, r.falls_total)
                             for r in results])


def results_json(results: Sequence[ExportResult]) -> list[dict[str, object]]:
    return [{
        "run_id": r.run_id, "method": r.method, "file": str(r.path), "size_kb": r.run.size_kb,
        "ms_per_window": r.run.ms_per_window, "falls_total": r.falls_total,
        "accuracy_float": r.marks.accuracy_float, "accuracy_int8": r.marks.accuracy_int8,
        "label_agreement": r.marks.label_agreement, "fall_agreement": r.marks.fall_agreement,
        "falls_found_float": r.marks.falls_found_float,
        "falls_found_int8": r.marks.falls_found_int8,
        "float_compute_ops": list(r.marks.float_compute_ops),
        "d53_marks_pass": r.marks.ops_pass and r.marks.accuracy_pass and r.marks.agreement_pass,
        "gate_pass": r.marks.gate_pass,
    } for r in results]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, action="append", required=True,
                        help="a run's TOML config (repeat for several runs)")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, help="default: <data-root>/runs")
    parser.add_argument("--out", type=Path, default=Path("results/f3b/int8.md"))
    parser.add_argument("--models-dir", type=Path, help="default: <data-root>/models")
    parser.add_argument("--method", choices=("qat", "ptq"), default="qat")
    parser.add_argument("--epochs", type=int, default=3, help="QAT passes over the training data")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    runs_dir = args.runs_dir or args.data_root / "runs"
    models_dir = args.models_dir or args.data_root / "models"
    results: list[ExportResult] = []
    for config in args.config:
        cfg = load_config(config)
        torch.manual_seed(cfg.seed)
        results.append(export_run(cfg, args.data_root, runs_dir / cfg.run_id / "best.pt",
                                  models_dir, args.method, args.epochs))
        logger.info("%s: gate %s", cfg.run_id, _verdict(results[-1].marks.gate_pass))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(results_markdown(results), encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps(results_json(results), indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
