"""Pose model over videos or PNG-sequence folders → per-frame keypoints as .npz (methods.md §6.1 step 1).

Sources: a video file, a directory of videos (searched recursively), or directories that hold an image
sequence such as UR Fall's `data/urfall/rgb/<sequence>/` (see scripts/prepare_urfall.py).

Output per source: <out>/<stem>.npz with ts[T] (s), frame_id[T] (the number in the file name, or 1-based
index), keypoints[T, 17, 3] (x, y, conf; zeros when no person), bbox[T, 4], person_present[T], fps.
Frames are read, inferred and discarded; nothing but keypoints is written (hard rule 2). Multi-person
frames keep the highest-confidence detection (MVP is single-person).

    python scripts/extract_keypoints.py data/urfall/rgb --out data/urfall/keypoints
    python scripts/extract_keypoints.py clip.mp4 --out data/self_recorded/keypoints --model yolov8n-pose.pt
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np

VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
IMAGE_EXT = {".png", ".jpg", ".jpeg"}
N_KPTS = 17  # COCO layout (YOLO-pose). MediaPipe (33) gets its own extractor if chosen in Sprint 1.
URFALL_FPS = 30.0  # frame rate stated on the UR Fall dataset page; applies to image sequences only
_TRAILING_INT = re.compile(r"(\d+)$")


def _image_files(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT)


def iter_sources(src: Path) -> list[Path]:
    """Video files plus every directory that directly contains an image sequence."""
    if src.is_file():
        return [src]
    videos = [p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXT]
    seq_dirs = [d for d in [src, *src.rglob("*")] if d.is_dir() and _image_files(d)]
    return sorted(set(videos) | set(seq_dirs))


def frame_id_of(path: Path, index: int) -> int:
    m = _TRAILING_INT.search(path.stem)
    return int(m.group(1)) if m else index + 1


def read_frames(source: Path, image_fps: float) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_id, ts, bgr_frame) from a video or an image-sequence directory."""
    import cv2  # noqa: PLC0415

    if source.is_dir():
        for i, f in enumerate(_image_files(source)):
            frame = cv2.imread(str(f))
            if frame is None:
                print(f"  unreadable image skipped: {f.name}", file=sys.stderr)
                continue
            yield frame_id_of(f, i), i / image_fps, frame
        return
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield i + 1, i / fps, frame
        i += 1
    cap.release()


def extract(source: Path, model, imgsz: int, device: str, image_fps: float) -> dict[str, np.ndarray]:
    ts, ids, kps, boxes, present = [], [], [], [], []
    for frame_id, t, frame in read_frames(source, image_fps):
        r = model.predict(frame, imgsz=imgsz, device=device, verbose=False)[0]
        if r.keypoints is not None and r.boxes is not None and len(r.boxes) > 0:
            best = int(r.boxes.conf.argmax())
            xy = r.keypoints.xy[best].cpu().numpy()  # [17, 2]
            conf = r.keypoints.conf[best].cpu().numpy()[:, None]  # [17, 1]
            kps.append(np.hstack([xy, conf]).astype(np.float32))
            boxes.append(r.boxes.xyxy[best].cpu().numpy().astype(np.float32))
            present.append(True)
        else:
            kps.append(np.zeros((N_KPTS, 3), np.float32))
            boxes.append(np.zeros(4, np.float32))
            present.append(False)
        ts.append(t)
        ids.append(frame_id)
    if source.is_dir():
        fps = image_fps
    elif len(ts) > 1 and ts[-1] > 0:
        fps = (len(ts) - 1) / ts[-1]
    else:
        fps = 0.0
    return {
        "ts": np.asarray(ts, np.float64),
        "frame_id": np.asarray(ids, np.int64),
        "keypoints": np.stack(kps) if kps else np.zeros((0, N_KPTS, 3), np.float32),
        "bbox": np.stack(boxes) if boxes else np.zeros((0, 4), np.float32),
        "person_present": np.asarray(present, bool),
        "fps": np.float64(fps),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("src", type=Path, help="video file, directory of videos, or image-sequence directories")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="yolo11n-pose.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--fps", type=float, default=URFALL_FPS, help="frame rate assumed for image sequences")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    from ultralytics import YOLO  # noqa: PLC0415

    sources = iter_sources(args.src)
    if not sources:
        print(f"no videos or image sequences under {args.src}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    for s in sources:
        dst = args.out / f"{s.stem if s.is_file() else s.name}.npz"
        if dst.exists() and not args.overwrite:
            print(f"skip {s.name} (exists)")
            continue
        data = extract(s, model, args.imgsz, args.device, args.fps)
        np.savez_compressed(dst, **data)
        n = len(data["ts"])
        print(f"{s.name}: {n} frames, person in {int(data['person_present'].sum())} -> {dst.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
