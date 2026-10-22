"""Fetch UP-Fall one clip at a time: download, check, extract, verify, delete (D-54).

UP-Fall is about 330 GB of PNG frames (9.3 h of footage, our estimate from the first zip,
2026-10-20) against about 39 GB free on the laptop, so only one trial-camera zip is on disk
at a time (PO 2026-10-20). A download is kept only if it is a complete, readable zip (a
quota-exceeded page from Google Drive is not); a zip is deleted only after every pose
model's keypoints are saved and hold all of its frames (extract_dataset.run). A re-run
skips finished clips and picks up a zip left by an interrupted run. A worker stops after
REFUSALS_TO_STOP downloads in a row come back as a page instead of a zip: Google Drive
answered every download with "Quota exceeded" from 20:51 on 2026-10-20, and carrying on
only sends more refused requests (D-62). The refusals hit downloads without a Google
sign-in only (the PO's test, 2026-10-21), so with --no-download the zips the PO downloaded
by hand, signed in, are imported from --raw-dir with the same checks (D-62).

Usage: python -m ally_fall.fetch_upfall --data-root ~/ally-data --page PAGE.html --subjects 1
       [--raw-dir DIR --no-download]
"""

from __future__ import annotations

import argparse
import http.client
import logging
import os
import shutil
import sys
import time
import urllib.request
import zipfile
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from ally_fall.build import keypoint_path
from ally_fall.datasets import VideoItem, up_fall
from ally_fall.extract_dataset import MODEL_FILES, ExtractReport, run
from ally_fall.omnifall import video_identities

logger = logging.getLogger(__name__)

DRIVE_URL = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"
ZIP_SIGNATURE = b"PK\x03\x04"
MARGIN_BYTES = 15 * 1024**3  # the F1 plan's free-space margin on the Windows drive
CHUNK_BYTES = 1024 * 1024
REFUSALS_TO_STOP = 3


class Refused(ValueError):
    """The server sent a page (quota or error) instead of the zip."""


@dataclass
class FetchReport:
    done: list[str] = field(default_factory=list)
    already_done: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (video_id, reason)
    missing: list[str] = field(default_factory=list)  # no zip and no downloader
    stopped: str | None = None


def check_download(path: Path, expected_size: int | None) -> None:
    """Raise ValueError unless the file is a complete zip whose members all pass their CRC."""
    with open(path, "rb") as file:
        if file.read(len(ZIP_SIGNATURE)) != ZIP_SIGNATURE:
            raise Refused(f"{path.name} is not a zip (quota or error page?)")
    size = path.stat().st_size
    if expected_size is not None and size != expected_size:
        raise ValueError(f"{path.name}: {size} bytes, the server sent {expected_size}")
    try:
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
    except (zipfile.BadZipFile, zlib.error, OSError) as error:
        raise ValueError(f"{path.name} is damaged: {error}") from error
    if bad is not None:
        raise ValueError(f"{path.name} is damaged at {bad}")


def download_drive(drive_id: str, target: Path) -> None:  # pragma: no cover - network
    """Download one Drive file to `target` through a .part file; checked before it is renamed."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    try:
        with urllib.request.urlopen(DRIVE_URL.format(id=drive_id), timeout=120) as response:
            length = response.headers.get("Content-Length")
            with open(part, "wb") as file:
                shutil.copyfileobj(response, file, CHUNK_BYTES)
        check_download(part, int(length) if length else None)
    except Refused:
        part.unlink(missing_ok=True)
        raise
    except (OSError, ValueError, http.client.HTTPException) as error:  # incl. a dropped stream
        part.unlink(missing_ok=True)
        raise ValueError(str(error)) from error
    os.replace(part, target)


def plan_clips(
    links: Mapping[str, str], omnifall_dir: Path, raw_dir: Path, subjects: Sequence[int]
) -> list[tuple[VideoItem, str]]:
    """OmniFall's clips for the chosen subjects, each with its zip path and Drive file id."""
    identity = video_identities(omnifall_dir / "labels" / "up_fall.csv")
    clips: list[tuple[VideoItem, str]] = []
    for video_id, (subject, _cam) in identity.items():
        if subject not in subjects:
            continue
        if video_id not in links:
            raise ValueError(f"no download link on the HAR-UP page for {video_id}")
        zip_path = raw_dir / (video_id.rsplit("/", 1)[1] + ".zip")
        clips.append((up_fall.item_of(zip_path, identity), links[video_id]))
    return sorted(clips, key=lambda clip: up_fall.sort_key(clip[0]))


def stray_files(raw_dir: Path, links: Mapping[str, str]) -> list[str]:
    """Files in `raw_dir` that are not the zip of any UP-Fall clip: a browser's `name (1).zip`
    copy, an unfinished `.crdownload`, anything else. Reported, never guessed at."""
    if not raw_dir.is_dir():
        return []
    expected = {video_id.rsplit("/", 1)[1] + ".zip" for video_id in links}
    return sorted(file.name for file in raw_dir.iterdir() if file.name not in expected)


def fetch(
    clips: Sequence[tuple[VideoItem, str]],
    data_root: Path,
    *,
    download: Callable[[str, Path], None] | None,
    process: Callable[[Sequence[VideoItem]], ExtractReport],
    free_bytes: Callable[[], int],
    pause_s: float,
    models: Sequence[str] = tuple(MODEL_FILES),
) -> FetchReport:
    """For each clip not yet extracted: check free space, download, then extract and delete.

    Without a downloader, a clip whose zip is not in place is listed as missing. A zip that
    was already in place (left by an interrupted run or downloaded by hand) is checked
    first; a bad one is kept and reported.
    """
    report = FetchReport()
    refused = 0
    for item, drive_id in clips:
        outputs = [
            keypoint_path(data_root, m, up_fall.DATASET, item.video_id, item.cam) for m in models
        ]
        if all(path.exists() for path in outputs):
            report.already_done.append(item.video_id)
            continue
        if item.source.exists():
            try:
                check_download(item.source, None)
            except ValueError as error:
                logger.error("zip of %s is not usable: %s", item.video_id, error)
                report.failed.append((item.video_id, f"zip: {error}"))
                continue
        elif download is None:
            report.missing.append(item.video_id)
            continue
        else:
            free = free_bytes()
            if free < MARGIN_BYTES:
                report.stopped = f"stopped before {item.video_id}: {free / 1024**3:.1f} GB free"
                logger.error(report.stopped)
                break
            try:
                download(drive_id, item.source)
            except ValueError as error:
                logger.error("download of %s failed: %s", item.video_id, error)
                report.failed.append((item.video_id, f"download: {error}"))
                refused = refused + 1 if isinstance(error, Refused) else refused
                if refused >= REFUSALS_TO_STOP:
                    report.stopped = f"stopped after {refused} refused downloads in a row"
                    logger.error(report.stopped)
                    break
                continue
            finally:
                time.sleep(pause_s)
            refused = 0
        extracted = process([item])
        report.failed.extend(extracted.failed)
        report.done.extend(extracted.done)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Download and extract UP-Fall clip by clip")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--page", type=Path, required=True, help="saved HAR-UP page (HTML)")
    parser.add_argument("--subjects", type=int, nargs="+", required=True)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--free-space-path", type=Path, default=Path("/mnt/c"),
                        help="the drive whose free space limits the run (Windows C: from WSL)")
    parser.add_argument("--pause", type=float, default=2.0, help="seconds between downloads")
    parser.add_argument("--raw-dir", type=Path, default=None,
                        help="folder of the zips (default DATA_ROOT/raw/up_fall)")
    parser.add_argument("--no-download", action="store_true",
                        help="never contact Drive; import the zips already in --raw-dir")
    args = parser.parse_args(argv)
    data_root = args.data_root.expanduser()
    models_dir = args.models_dir or data_root / "models"
    models = {name: models_dir / file for name, file in MODEL_FILES.items()}
    links = up_fall.camera_links(args.page.read_text(encoding="utf-8"))
    raw_dir = args.raw_dir.expanduser() if args.raw_dir else data_root / "raw" / "up_fall"
    clips = plan_clips(links, data_root / "omnifall", raw_dir, args.subjects)
    strays = stray_files(raw_dir, links)
    for name in strays:
        logger.error("not a UP-Fall clip zip, left alone: %s", name)
    pending = [
        item for item, _ in clips
        if not all(keypoint_path(data_root, m, up_fall.DATASET, item.video_id, item.cam).exists()
                   for m in models)
    ]
    if pending and not all(path.exists() for path in models.values()):
        logger.error("pose model files missing in %s", models_dir)
        return 1
    report = fetch(
        clips,
        data_root,
        download=None if args.no_download else download_drive,
        process=partial(run, models=models, data_root=data_root, dataset=up_fall.DATASET,
                        delete=True),
        free_bytes=lambda: shutil.disk_usage(args.free_space_path).free,
        pause_s=args.pause,
    )
    logger.info(
        "%d done, %d already done, %d failed, %d missing, %d stray files%s",
        len(report.done), len(report.already_done), len(report.failed), len(report.missing),
        len(strays), f"; {report.stopped}" if report.stopped else "",
    )
    for video_id, reason in report.failed:
        logger.info("failed: %s (%s)", video_id, reason)
    return 1 if report.failed or report.stopped or strays else 0


if __name__ == "__main__":
    sys.exit(main())
