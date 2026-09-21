"""Run a pose model over video files and store per-frame keypoints as .npz (methods.md §6.1 step 1).

Output per video: <out>/<stem>.npz with ts[T] (s), keypoints[T, 17, 3] (x, y, conf; zeros when no person),
bbox[T, 4], person_present[T]. Frames are read, inferred and discarded; nothing but keypoints is written
(hard rule 2). Multi-person frames keep the highest-confidence detection (MVP is single-person).

    python scripts/extract_keypoints.py data/urfall/rgb --out data/urfall/keypoints
    python scripts/extract_keypoints.py clip.mp4 --out data/self_recorded/keypoints --model yolov8n-pose.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

VIDEO_EXT = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
N_KPTS = 17  # COCO layout (YOLO-pose). MediaPipe (33) gets its own extractor if chosen in Sprint 1.


def iter_videos(src: Path) -> list[Path]:
    if src.is_file():
        return [src]
    return sorted(p for p in src.rglob("*") if p.suffix.lower() in VIDEO_EXT)


def extract(video: Path, model, imgsz: int, device: str) -> dict[str, np.ndarray]:
    import cv2  # noqa: PLC0415

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ts, kps, boxes, present = [], [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
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
        ts.append(i / fps)
        i += 1
    cap.release()
    return {
        "ts": np.asarray(ts, np.float64),
        "keypoints": np.stack(kps) if kps else np.zeros((0, N_KPTS, 3), np.float32),
        "bbox": np.stack(boxes) if boxes else np.zeros((0, 4), np.float32),
        "person_present": np.asarray(present, bool),
        "fps": np.float64(fps),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("src", type=Path, help="video file or directory (searched recursively)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="yolo11n-pose.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    from ultralytics import YOLO  # noqa: PLC0415

    videos = iter_videos(args.src)
    if not videos:
        print(f"no videos under {args.src}", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    for v in videos:
        dst = args.out / f"{v.stem}.npz"
        if dst.exists() and not args.overwrite:
            print(f"skip {v.name} (exists)")
            continue
        data = extract(v, model, args.imgsz, args.device)
        np.savez_compressed(dst, **data)
        n = len(data["ts"])
        print(f"{v.name}: {n} frames, person in {int(data['person_present'].sum())} -> {dst.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
