"""UP-Fall (HAR-UP site, sites.google.com/up.edu.mx/har-up; no licence named, D-54 risk 4).

One zip of PNG frames per trial and camera, `Subject{s}Activity{a}Trial{t}Camera{c}.zip`.
OmniFall names it `Subject{s}/Activity{a}/Trial{t}/Subject{s}Activity{a}Trial{t}Camera{c}`
(checked 2026-10-20: the page's 1,118 camera links match OmniFall's 1,118 clips).
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping
from pathlib import Path

from ally_fall.datasets import VideoItem
from ally_fall.omnifall import video_identities

DATASET = "up_fall"
_ZIP = re.compile(r"^Subject(\d+)Activity(\d+)Trial(\d+)Camera(\d+)\.zip$")
_TRIAL = re.compile(r"<h5>(Subject\d+Activity\d+Trial\d+)</h5>")
_CAMERA_LINK = re.compile(r"uc\?id=([\w-]+)&export=download\">Camera([12])</a>")


def video_id_of(trial_camera: str) -> str:
    """`Subject1Activity1Trial1Camera1` -> OmniFall's path for it."""
    match = _ZIP.match(trial_camera + ".zip")
    if match is None:
        raise ValueError(f"unexpected UP-Fall name: {trial_camera}")
    s, a, t, _ = match.groups()
    return f"Subject{s}/Activity{a}/Trial{t}/{trial_camera}"


def sort_key(item: VideoItem) -> tuple[int, ...]:
    """Subject, activity, trial and camera as numbers (text order would put 16 before 2)."""
    return tuple(int(n) for n in re.findall(r"\d+", item.video_id.rsplit("/", 1)[1]))


def item_of(file: Path, identity: Mapping[str, tuple[int, int]]) -> VideoItem:
    """The OmniFall identity of one downloaded zip; its subject and camera must match the table."""
    match = _ZIP.match(file.name)
    if match is None:
        raise ValueError(f"unexpected UP-Fall file: {file}")
    video_id = video_id_of(file.name.removesuffix(".zip"))
    if video_id not in identity:
        raise ValueError(f"no OmniFall labels for {video_id} ({file})")
    subject, cam = identity[video_id]
    if (subject, cam) != (int(match.group(1)), int(match.group(4))):
        raise ValueError(f"{file.name}: OmniFall gives subject {subject}, camera {cam}")
    return VideoItem(source=file, video_id=video_id, subject=subject, cam=cam)


def items(raw_dir: Path, omnifall_dir: Path) -> list[VideoItem]:
    """Every downloaded zip, in subject, activity, trial and camera order."""
    identity = video_identities(omnifall_dir / "labels" / "up_fall.csv")
    found = [item_of(file, identity) for file in raw_dir.rglob("*.zip")]
    return sorted(found, key=sort_key)


def camera_links(page_html: str) -> dict[str, str]:
    """OmniFall path -> Google Drive file id, for every Camera1/Camera2 link on the HAR-UP page.

    The page embeds its download table HTML-escaped twice, so it is unescaped first.
    """
    text = html.unescape(html.unescape(page_html))
    parts = _TRIAL.split(text)
    links: dict[str, str] = {}
    for trial, body in zip(parts[1::2], parts[2::2], strict=True):
        for drive_id, cam in _CAMERA_LINK.findall(body):
            video_id = video_id_of(f"{trial}Camera{cam}")
            if links.get(video_id, drive_id) != drive_id:
                raise ValueError(f"{trial}Camera{cam} is listed with two different files")
            links[video_id] = drive_id
    return links
