"""Tests for the INT8 export (F3b Task 5, D-53 marks, D-68 decision 3 gate).

The mark and report logic needs no converter. The tests marked `converter` run the real
LiteRT Torch conversion on a small made-up TCN; they are skipped where the converter is not
installed (CI, D-52: the pins stay out of CI, the PO's local run runs them).
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch

import ally_fall.export as export
from ally_fall.models import TCN

try:
    import litert_torch  # noqa: F401
    HAVE_CONVERTER = True
except ImportError:
    HAVE_CONVERTER = False

converter = pytest.mark.skipif(not HAVE_CONVERTER, reason="LiteRT Torch is not installed")
FALL = (3, 4)  # class ids that count as a fall (fall, fallen)
N_CLASSES = 5


def _marks(float_pred: list[int], int8_pred: list[int], y: list[int], *, float_found: int = 56,
           int8_found: int = 56, ops: tuple[str, ...] = ()) -> export.Marks:
    return export.compute_marks(np.array(float_pred), np.array(int8_pred), np.array(y), FALL,
                                float_found, int8_found, ops)


def test_marks_use_the_d53_values() -> None:
    assert export.MAX_ACCURACY_DROP == 0.02
    assert export.MIN_AGREEMENT == 0.98
    assert export.MAX_FEWER_FALLS == 2


def test_agreement_and_accuracy_are_computed_on_given_windows() -> None:
    y = [0, 1, 2, 3]
    m = _marks(float_pred=[0, 1, 2, 3], int8_pred=[0, 1, 0, 3], y=y)
    assert m.accuracy_float == 1.0
    assert m.accuracy_int8 == 0.75
    assert m.accuracy_drop == pytest.approx(0.25)
    assert m.label_agreement == 0.75


def test_fall_decision_agreement_is_reported_separately() -> None:
    # fall (3) versus fallen (4) is a different label but the same fall decision
    m = _marks(float_pred=[3, 3, 0, 0], int8_pred=[4, 3, 0, 1], y=[3, 3, 0, 0])
    assert m.label_agreement == 0.5
    assert m.fall_agreement == 1.0
    # a flip across the fall boundary lowers the fall agreement
    m = _marks(float_pred=[3, 0, 0, 0], int8_pred=[0, 0, 0, 0], y=[3, 0, 0, 0])
    assert m.fall_agreement == 0.75


def test_the_three_d53_marks_and_the_gate_are_separate_verdicts() -> None:
    clean = _marks([3, 0] * 50, [3, 0] * 50, [3, 0] * 50)
    assert clean.ops_pass and clean.accuracy_pass and clean.agreement_pass and clean.gate_pass
    flipped = _marks([0] * 100, [0] * 97 + [1] * 3, [0] * 100)  # 97% label agreement
    assert not flipped.agreement_pass
    assert flipped.gate_pass  # nothing crossed the fall boundary: the gate is about the decision
    float_op = _marks([0] * 10, [0] * 10, [0] * 10, ops=("MEAN",))
    assert not float_op.ops_pass


def test_gate_needs_98_percent_fall_agreement_and_at_most_one_fall_lost() -> None:
    same = [3, 0] * 50
    assert _marks(same, same, same, float_found=56, int8_found=55).gate_pass  # one fewer: allowed
    assert not _marks(same, same, same, float_found=56, int8_found=54).gate_pass  # two fewer
    assert _marks(same, same, same, float_found=56, int8_found=58).gate_pass  # more is fine
    truth = [3] * 50 + [0] * 50
    two_lost = [0, 0] + [3] * 48 + [0] * 50  # 2 of 100 windows cross the fall boundary
    assert _marks(truth, two_lost, truth).fall_agreement == 0.98
    assert _marks(truth, two_lost, truth).gate_pass  # on the mark
    three_lost = [0, 0, 0] + [3] * 47 + [0] * 50
    assert _marks(truth, three_lost, truth).fall_agreement == 0.97
    assert not _marks(truth, three_lost, truth).gate_pass


def test_marks_refuse_empty_or_mismatched_windows() -> None:
    with pytest.raises(ValueError, match="same windows"):
        _marks([0, 1], [0], [0, 1])
    with pytest.raises(ValueError, match="no windows"):
        _marks([], [], [])


def test_report_names_the_marks_the_gate_and_what_is_not_measured() -> None:
    m = _marks([3, 0] * 50, [3, 0] * 50, [3, 0] * 50, float_found=56, int8_found=55)
    run = export.TfliteRun(size_kb=61.5, ms_per_window=0.04,
                           ops={"CONV_2D": ["int8"]}, float_compute_ops=())
    text = export.report_markdown("tcn-omnifall16-b0.999-s0", "qat", m, run, falls_total=57)
    assert "tcn-omnifall16-b0.999-s0" in text and "QAT" in text
    assert "55 of 57" in text and "56 of 57" in text
    assert "PASS" in text
    assert "61.5" in text
    assert "phone" in text.lower()  # phone speed is not measured here (Spike 2)


def test_batches_cover_every_window_once_per_epoch() -> None:
    order = list(export.epoch_batches(10, 4, np.random.default_rng(0)))
    assert sorted(i for batch in order for i in batch) == list(range(10))
    assert [len(b) for b in order] == [4, 4, 2]


def test_a_batch_of_one_is_dropped_because_batch_norm_needs_two() -> None:
    sizes = [len(b) for b in export.epoch_batches(9, 4, np.random.default_rng(0))]
    assert sizes == [4, 4]


def _windows(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0, 1, (n, 3, 30, 33)).astype(np.float32)


def _labels(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, N_CLASSES, n).astype(np.int64)


def _batches(n: int = 128, size: int = 32) -> export.Batches:
    x, y = _windows(n), _labels(n)

    def source() -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for start in range(0, n, size):
            yield x[start:start + size], y[start:start + size]

    return source


@converter
def test_int8_file_has_no_float_compute_ops(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = TCN(N_CLASSES, hidden=8)
    path = tmp_path / "tcn.tflite"
    export.convert_int8(model, "qat", _batches(), np.ones(N_CLASSES, np.float32), path, epochs=1)
    run = export.run_tflite(path, _windows(8, 1))
    assert path.exists()
    assert run.preds.shape == (8,)
    assert run.float_compute_ops == (), run.ops


@converter
def test_size_and_speed_are_recorded(tmp_path: Path) -> None:
    torch.manual_seed(0)
    path = tmp_path / "tcn.tflite"
    export.convert_int8(TCN(N_CLASSES, hidden=8), "ptq", _batches(), np.ones(N_CLASSES, np.float32),
                        path, epochs=0)
    run = export.run_tflite(path, _windows(5, 2))
    assert run.size_kb == pytest.approx(path.stat().st_size / 1024)
    assert run.ms_per_window > 0
    assert "CONV_2D" in run.ops


@converter
def test_int8_answers_follow_the_float_model_on_the_same_windows(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = TCN(N_CLASSES, hidden=8)
    x, y = _windows(200, 3), _labels(200, 3)

    def source() -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for start in range(0, 200, 50):
            yield x[start:start + 50], y[start:start + 50]

    # a briefly fitted model, so its answers are not all one class
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(30):
        for xb, yb in source():
            opt.zero_grad()
            torch.nn.functional.cross_entropy(model(torch.from_numpy(xb)),
                                              torch.from_numpy(yb)).backward()
            opt.step()
    path = tmp_path / "tcn.tflite"
    export.convert_int8(model, "qat", source, np.ones(N_CLASSES, np.float32), path, epochs=2)
    run = export.run_tflite(path, x)
    float_pred = model.eval()(torch.from_numpy(x)).argmax(1).numpy()
    # random inputs; the real mark is on real windows
    assert (run.preds == float_pred).mean() >= 0.8


def test_convert_int8_refuses_an_unknown_method(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="method"):
        export.convert_int8(TCN(N_CLASSES, hidden=8), "dynamic", _batches(),
                            np.ones(N_CLASSES, np.float32), tmp_path / "x.tflite", epochs=1)
