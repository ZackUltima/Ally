from pathlib import Path

import numpy as np

from ally_fall.data import load_split, person_of, window_tensor


def _built(path: Path) -> Path:
    """Two videos of 40 and 35 ticks; feature value = global tick index (x), -index (y), 1 (vis)."""
    lengths = np.array([40, 35])
    starts = np.array([0, 40])
    ticks = np.arange(lengths.sum(), dtype=np.float32)
    features = np.zeros((lengths.sum(), 33, 3), dtype=np.float32)
    features[:, :, 0] = ticks[:, None]
    features[:, :, 1] = -ticks[:, None]
    features[:, :, 2] = 1.0
    labels = np.zeros(lengths.sum(), dtype=np.int64)
    labels[70:] = 1
    window_end = np.array([29, 31, 39, 69, 74])
    np.savez(path, features=features, labels=labels, video_start=starts, video_len=lengths,
             subject=np.array([1, 1]), dataset=np.array(["GMDCSA24", "le2i"]),
             video_id=np.array(["a", "b"]), window_end=window_end, y=labels[window_end])
    return path


def test_window_tensor_layout(tmp_path: Path) -> None:
    split = load_split(_built(tmp_path / "train.npz"), np.arange(16))
    w = window_tensor(split, np.array([1]))
    assert w.shape == (1, 3, 30, 33) and w.dtype == np.float32
    assert w[0, 0, :, 0].tolist() == list(range(2, 32))  # ticks 2..31, channel 0 = x
    assert w[0, 1, -1, 5] == -31.0 and w[0, 2, 0, 0] == 1.0  # channel 1 = y, 2 = visibility


def test_load_split_maps_labels(tmp_path: Path) -> None:
    mapping = np.zeros(16, dtype=np.int64)
    mapping[1] = 4
    split = load_split(_built(tmp_path / "train.npz"), mapping)
    assert split.y.tolist() == [0, 0, 0, 0, 4]


def test_person_of(tmp_path: Path) -> None:
    split = load_split(_built(tmp_path / "train.npz"), np.arange(16))
    person = person_of(split)
    assert person[0] == person[1] == person[2]  # GMDCSA24 subject 1
    assert person[3] == person[4] and person[3] != person[0]  # Le2i subject 1 is someone else
