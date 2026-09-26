"""Person-detection rate of extracted UR Fall keypoints, split by label (check after extract_keypoints.py).

Joins data/urfall/keypoints/<seq>.npz (frame_id) to data/urfall/labels.csv (sequence, frame, label) and prints
the detection rate per label class and the worst sequences. Frames without a label row are reported separately
and must not be treated as ADL (UR Fall annotates only frames where its own pipeline found the person).

    python scripts/urfall_detection_rate.py [--keypoints data/urfall/keypoints] [--labels <labels.csv>]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

LABEL_NAME = {-1: "not lying (-1)", 0: "falling (0)", 1: "lying (1)"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--keypoints", type=Path, default=Path("data/urfall/keypoints"))
    p.add_argument("--labels", type=Path, default=Path("data/urfall/labels.csv"))
    p.add_argument("--worst", type=int, default=5)
    args = p.parse_args()

    labels: dict[tuple[str, int], int] = {}
    with args.labels.open(newline="") as f:
        for r in csv.DictReader(f):
            labels[(r["sequence"], int(r["frame"]))] = int(r["label"])

    by_label: dict[str, list[bool]] = defaultdict(list)
    per_seq: list[tuple[float, str, int, int]] = []
    files = sorted(args.keypoints.glob("*.npz"))
    if not files:
        print(f"no .npz under {args.keypoints}")
        return 1
    for npz in files:
        d = np.load(npz)
        present = d["person_present"]
        for fid, ok in zip(d["frame_id"].tolist(), present.tolist(), strict=True):
            lab = labels.get((npz.stem, fid))
            by_label[LABEL_NAME.get(lab, "unlabelled") if lab is not None else "unlabelled"].append(ok)
        per_seq.append((float(present.mean()), npz.stem, int(present.sum()), len(present)))

    total = sum(len(v) for v in by_label.values())
    print(f"{len(files)} sequences, {total} frames")
    for name, v in sorted(by_label.items()):
        print(f"  {name:<16} {sum(v):6d}/{len(v):<6d} detected  ({100 * sum(v) / len(v):5.1f}%)")
    kinds = defaultdict(list)
    for rate, seq, _, _ in per_seq:
        kinds[seq.split("-")[0]].append(rate)
    for k, v in sorted(kinds.items()):
        print(f"  {k:<5} sequences: mean rate {100 * np.mean(v):5.1f}%  min {100 * min(v):5.1f}%")
    print(f"worst {args.worst}:")
    for rate, seq, n_ok, n in sorted(per_seq)[: args.worst]:
        print(f"  {seq:<8} {n_ok:4d}/{n:<4d} ({100 * rate:5.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
