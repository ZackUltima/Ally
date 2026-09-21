"""Download and lay out UR Fall (CC BY-NC-SA 4.0) under data/urfall/ (methods.md §6.1 step 1).

Source: http://fenix.ur.edu.pl/~mkepski/ds/uf.html — 30 fall and 40 ADL sequences, camera 0 RGB, ~58 MB per
sequence as a zip of PNG frames (≈ 4 GB for everything). Only the RGB frames of cam0 are fetched; depth and
accelerometer files are not needed for a pose-based detector.

Layout written
    data/urfall/
    ├── annotations/urfall-cam0-falls.csv, urfall-cam0-adls.csv   (the authors' per-frame files, verbatim)
    ├── rgb/<sequence>/<sequence>-cam0-rgb-NNN.png                  (frames; git-ignored, never committed)
    ├── labels.csv        sequence,frame,label,kind   label per the dataset page: 1 = lying on the ground,
    │                     0 = "temporary pose, when person is falling" (the authors exclude these from
    │                     classification), -1 = not lying; kind = fall | adl. `frame` is the PNG file-name
    │                     number (ADL files skip some frames — join by id, never by position)
    ├── subjects.csv      sequence,subject — filled BY HAND for leave-one-subject-out CV. The dataset does
    │                     not publish a sequence → subject map; do not guess it (decisions.md, Sprint 1 notes)
    └── manifest.json     what is present, frame counts, source, licence

    python scripts/prepare_urfall.py --dry-run                # list what would be fetched
    python scripts/prepare_urfall.py --only falls --limit 1   # one sequence, to test the pipeline
    python scripts/prepare_urfall.py                          # everything (~4 GB, resumable: done ones skip)

Then: python scripts/extract_keypoints.py data/urfall/rgb --out data/urfall/keypoints
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import urllib.parse
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

BASE_URL = "https://fenix.ur.edu.pl/mkepski/ds/data/"
ALLOWED_HOST = "fenix.ur.edu.pl"
LICENCE = "CC BY-NC-SA 4.0 — Kwolek & Kepski, UR Fall Detection Dataset"
N_FALLS, N_ADLS = 30, 40
CAM = 0
CHUNK = 1 << 20
REPO_ROOT = Path(__file__).resolve().parent.parent


def sequences(only: str) -> list[tuple[str, str]]:
    falls = [(f"fall-{i:02d}", "fall") for i in range(1, N_FALLS + 1)]
    adls = [(f"adl-{i:02d}", "adl") for i in range(1, N_ADLS + 1)]
    return {"falls": falls, "adls": adls, "all": falls + adls}[only]


def _checked_url(name: str) -> str:
    url = urllib.parse.urljoin(BASE_URL, name)
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or parts.hostname != ALLOWED_HOST:
        raise ValueError(f"refusing to fetch outside https://{ALLOWED_HOST}: {url}")
    return url


def download(name: str, dst: Path, *, quiet: bool = False) -> Path:
    """Fetch BASE_URL/name to dst atomically (.part then rename). Existing files are kept."""
    if dst.exists():
        return dst
    url = _checked_url(name)  # https + host allow-list; this is what the S310 waivers below rely on
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "ally-prepare-urfall/1.0"})  # noqa: S310
    with urllib.request.urlopen(req, timeout=60) as resp, part.open("wb") as fh:  # noqa: S310
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            buf = resp.read(CHUNK)
            if not buf:
                break
            fh.write(buf)
            done += len(buf)
            if not quiet and total and done % (8 * CHUNK) < CHUNK:
                print(f"\r  {name}: {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    if not quiet:
        print(f"\r  {name}: {done / 1e6:.1f} MB done" + " " * 20)
    part.replace(dst)
    return dst


def extract_frames(zip_path: Path, out_dir: Path) -> int:
    """Extract the PNG members of a UR Fall zip flat into out_dir (members are checked to stay inside it)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir() or not member.filename.lower().endswith(".png"):
                continue
            target = (out_dir / Path(member.filename).name).resolve()
            if out_dir.resolve() not in target.parents:
                raise RuntimeError(f"zip member escapes {out_dir}: {member.filename}")
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    return n


def write_labels(annotations: Path, out_csv: Path, present: set[str]) -> int:
    """Merge the authors' CSVs into sequence,frame,label,kind (first three columns only; the remaining
    columns are their depth-based features and stay in annotations/)."""
    rows = 0
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sequence", "frame", "label", "kind"])
        for kind, fname in (("fall", f"urfall-cam{CAM}-falls.csv"), ("adl", f"urfall-cam{CAM}-adls.csv")):
            with (annotations / fname).open(newline="", encoding="utf-8") as src:
                for rec in csv.reader(src):
                    if len(rec) < 3 or not rec[0]:
                        continue
                    seq, frame, label = rec[0].strip(), int(rec[1]), int(rec[2])
                    if seq in present:
                        w.writerow([seq, frame, label, kind])
                        rows += 1
    return rows


def write_subjects_template(path: Path, seqs: list[str]) -> bool:
    """sequence,subject with subject left blank; never overwrites a hand-filled file."""
    if path.exists():
        return False
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sequence", "subject"])
        for s in seqs:
            w.writerow([s, ""])
    return True


def subjects_missing(path: Path) -> bool:
    with path.open(newline="", encoding="utf-8") as fh:
        return any(not (r.get("subject") or "").strip() for r in csv.DictReader(fh))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dest", type=Path, default=REPO_ROOT / "data" / "urfall")
    p.add_argument("--only", choices=["falls", "adls", "all"], default="all")
    p.add_argument("--limit", type=int, default=None, help="first N sequences of the selection")
    p.add_argument("--dry-run", action="store_true", help="print the plan; fetch nothing")
    p.add_argument("--keep-zips", action="store_true", help="keep the downloaded zips under dest/zips/")
    args = p.parse_args()

    selection = sequences(args.only)
    if args.limit is not None:
        selection = selection[: args.limit]
    dest: Path = args.dest
    rgb, zips, annotations = dest / "rgb", dest / "zips", dest / "annotations"

    todo = [(s, k) for s, k in selection if not any((rgb / s).glob("*.png"))]
    print(f"UR Fall → {dest}  ({LICENCE})")
    print(f"  selection: {len(selection)} sequences, {len(todo)} to fetch (~{58 * len(todo) / 1000:.1f} GB)")
    if args.dry_run:
        for s, _ in todo:
            print(f"  would fetch {_checked_url(f'{s}-cam{CAM}-rgb.zip')}")
        return 0

    for fname in (f"urfall-cam{CAM}-falls.csv", f"urfall-cam{CAM}-adls.csv"):
        download(fname, annotations / fname)

    for seq, _ in todo:
        zname = f"{seq}-cam{CAM}-rgb.zip"
        zpath = download(zname, zips / zname)
        n = extract_frames(zpath, rgb / seq)
        print(f"  {seq}: {n} frames")
        if not args.keep_zips:
            zpath.unlink()
    if not args.keep_zips and zips.exists() and not any(zips.iterdir()):
        zips.rmdir()

    present = (
        {d.name for d in rgb.iterdir() if d.is_dir() and any(d.glob("*.png"))} if rgb.exists() else set()
    )
    n_rows = write_labels(annotations, dest / "labels.csv", present)
    all_seqs = [s for s, _ in sequences("all")]
    made = write_subjects_template(dest / "subjects.csv", all_seqs)
    manifest = {
        "prepared_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": BASE_URL,
        "licence": LICENCE,
        "camera": CAM,
        "sequences": {s: len(list((rgb / s).glob("*.png"))) for s in sorted(present)},
        "label_rows": n_rows,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  labels.csv: {n_rows} rows for {len(present)} sequences; manifest.json written")
    if made or subjects_missing(dest / "subjects.csv"):
        print(
            "  NOTE subjects.csv has blank subject IDs: fill it by hand before leave-one-subject-out CV "
            "(the dataset does not publish the mapping).",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
