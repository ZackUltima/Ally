"""FPS / VRAM profile of a pose model on the webcam (NFR-4; Sprint 1 "per-model VRAM table").

Runs the model on N frames at 640x480, reports mean FPS, p50/p95 inference latency and peak CUDA memory,
and appends a row to results/bench/pose_bench.csv (seed, model, git SHA, device). Frames stay in memory.

    python scripts/benchmark_fps.py                       # webcam 0, yolo11n-pose, 300 frames
    python scripts/benchmark_fps.py --model yolov8n-pose.pt --frames 600
    python scripts/benchmark_fps.py --source path/to/clip.mp4 --no-show
"""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ally.config import get_settings  # noqa: E402


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="yolo11n-pose.pt", help="Ultralytics pose weights (auto-downloaded)")
    p.add_argument("--source", default="0", help="webcam index or video path")
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0", help="'0' = first CUDA GPU, 'cpu' for CPU")
    p.add_argument("--half", action="store_true", help="FP16 inference (GPU only)")
    p.add_argument("--no-show", action="store_true", help="do not open a preview window")
    args = p.parse_args()

    import cv2  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from ultralytics import YOLO  # noqa: PLC0415

    s = get_settings()
    src: int | str = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW if isinstance(src, int) and sys.platform == "win32" else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, s.ally_capture_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, s.ally_capture_height)
    if not cap.isOpened():
        print(f"cannot open source {args.source!r}", file=sys.stderr)
        return 1

    use_cuda = args.device != "cpu" and torch.cuda.is_available()
    device = args.device if use_cuda else "cpu"
    half = bool(args.half and use_cuda)
    model = YOLO(args.model)
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    # warm-up (model load + first CUDA kernels are not representative)
    for _ in range(10):
        ok, frame = cap.read()
        if not ok:
            break
        model.predict(frame, imgsz=args.imgsz, device=device, half=half, verbose=False)

    infer_ms: list[float] = []
    t_start = time.perf_counter()
    n = 0
    while n < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        res = model.predict(frame, imgsz=args.imgsz, device=device, half=half, verbose=False)
        infer_ms.append((time.perf_counter() - t0) * 1000)
        n += 1
        if not args.no_show:
            cv2.imshow("ally bench (q to stop)", res[0].plot())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    wall = time.perf_counter() - t_start
    cap.release()
    cv2.destroyAllWindows()

    if n == 0:
        print("no frames read", file=sys.stderr)
        return 1

    fps = n / wall
    p50 = statistics.median(infer_ms)
    p95 = sorted(infer_ms)[int(0.95 * (len(infer_ms) - 1))]
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if use_cuda else 0.0
    gpu_name = torch.cuda.get_device_name(0) if use_cuda else "cpu"

    print(f"model={args.model} device={gpu_name} frames={n} imgsz={args.imgsz} half={half}")
    print(f"end-to-end FPS={fps:.1f}  inference p50={p50:.1f} ms  p95={p95:.1f} ms")
    print(f"peak VRAM={peak_mb:.0f} MB")
    print("PASS" if fps >= s.ally_target_fps else "FAIL", f"(target >= {s.ally_target_fps} FPS, hard rule 8)")

    out = s.ally_results_dir / "bench" / "pose_bench.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(
                [
                    "recorded_at",
                    "git_sha",
                    "model",
                    "device",
                    "imgsz",
                    "half",
                    "frames",
                    "fps",
                    "p50_ms",
                    "p95_ms",
                    "peak_vram_mb",
                ]
            )
        w.writerow(
            [
                datetime.now(UTC).isoformat(timespec="seconds"),
                git_sha(),
                args.model,
                gpu_name,
                args.imgsz,
                half,
                n,
                f"{fps:.2f}",
                f"{p50:.2f}",
                f"{p95:.2f}",
                f"{peak_mb:.0f}",
            ]
        )
    print(f"row appended to {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
