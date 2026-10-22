"""Tests for the UP-Fall fetch loop: one clip on disk at a time (D-54, PO 2026-10-20)."""

import zipfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from ally_fall.build import keypoint_path
from ally_fall.datasets import VideoItem
from ally_fall.extract_dataset import ExtractReport
from ally_fall.fetch_upfall import (
    MARGIN_BYTES,
    REFUSALS_TO_STOP,
    FetchReport,
    Refused,
    check_download,
    fetch,
    main,
    plan_clips,
    stray_files,
)
from ally_fall.keypoints import KeypointSeq, save_seq

ID = "Subject1/Activity1/Trial1/Subject1Activity1Trial1Camera1"
ID2 = "Subject1/Activity1/Trial1/Subject1Activity1Trial1Camera2"
TABLE = f"""path,label,start,end,subject,cam,dataset
{ID},8,0.0,2.89409,1,1,up_fall
{ID2},8,0.0,2.89409,1,2,up_fall
Subject2/Activity1/Trial1/Subject2Activity1Trial1Camera1,8,0.0,3.0,2,1,up_fall
"""
LINKS = {ID: "drive1", ID2: "drive2",
         "Subject2/Activity1/Trial1/Subject2Activity1Trial1Camera1": "drive3"}


def _zip(path: Path, frames: int = 2) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for i in range(frames):
            archive.writestr(f"2018-07-04T12_04_1{i}.png", b"x" * 10)
    return path


def _omnifall(tmp_path: Path) -> Path:
    labels = tmp_path / "omnifall" / "labels"
    labels.mkdir(parents=True)
    (labels / "up_fall.csv").write_text(TABLE)
    return tmp_path / "omnifall"


def _item(tmp_path: Path, video_id: str = ID, cam: int = 1) -> VideoItem:
    name = video_id.rsplit("/", 1)[1]
    return VideoItem(source=tmp_path / "raw" / f"{name}.zip", video_id=video_id, subject=1,
                     cam=cam)


def _save_both(data_root: Path, item: VideoItem) -> None:
    for model in ("lite", "full"):
        save_seq(KeypointSeq(
            dataset="up_fall", video_id=item.video_id, subject=1, cam=item.cam,
            pose_model=model, frame_times_s=np.zeros(1), xy=np.zeros((1, 33, 2), np.float32),
            vis=np.zeros((1, 33), np.float32),
        ), keypoint_path(data_root, model, "up_fall", item.video_id, item.cam))


class _Fake:
    """Records calls; the downloader writes a real zip, extraction deletes it like run()."""

    def __init__(self, fail_download: str | None = None, fail_extract: bool = False) -> None:
        self.downloads: list[str] = []
        self.fail_download = fail_download
        self.fail_extract = fail_extract

    def download(self, drive_id: str, target: Path) -> None:
        self.downloads.append(drive_id)
        if drive_id == self.fail_download:
            raise ValueError("not a zip (quota page?)")
        target.parent.mkdir(parents=True, exist_ok=True)
        _zip(target)

    def process(self, items: Sequence[VideoItem]) -> ExtractReport:
        report = ExtractReport()
        for item in items:
            if self.fail_extract:
                report.failed.append((item.video_id, "lite: 1 frames extracted, 2 in the source"))
            else:
                item.source.unlink()
                report.done.append(item.video_id)
        return report


def _run(
    tmp_path: Path, fake: _Fake, clips: list[tuple[VideoItem, str]], free: int = 10**12
) -> FetchReport:
    return fetch(clips, tmp_path, download=fake.download, process=fake.process,
                 free_bytes=lambda: free, pause_s=0.0)


def test_check_download_accepts_a_complete_zip(tmp_path: Path) -> None:
    clip = _zip(tmp_path / "c.zip")
    check_download(clip, clip.stat().st_size)
    check_download(clip, None)


def test_check_download_rejects_a_quota_page(tmp_path: Path) -> None:
    page = tmp_path / "c.zip"
    page.write_bytes(b"<!DOCTYPE html><html>Google Drive - Quota exceeded</html>")
    with pytest.raises(ValueError, match="not a zip"):
        check_download(page, None)


def test_check_download_rejects_a_short_file(tmp_path: Path) -> None:
    clip = _zip(tmp_path / "c.zip")
    with pytest.raises(ValueError, match="bytes"):
        check_download(clip, clip.stat().st_size + 1)


def test_check_download_rejects_a_damaged_member(tmp_path: Path) -> None:
    clip = tmp_path / "c.zip"
    with zipfile.ZipFile(clip, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("2018-07-04T12_04_10.png", b"abcdefghij")
    data = bytearray(clip.read_bytes())
    data[data.index(b"abcdefghij")] = ord("X")
    clip.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="damaged"):
        check_download(clip, None)


def test_fetch_downloads_then_processes_each_clip(tmp_path: Path) -> None:
    fake = _Fake()
    report = _run(tmp_path, fake, [(_item(tmp_path), "drive1"), (_item(tmp_path, ID2, 2), "d2")])
    assert fake.downloads == ["drive1", "d2"]
    assert report.done == [ID, ID2] and report.failed == [] and report.stopped is None
    assert list((tmp_path / "raw").iterdir()) == []


def test_fetch_skips_clips_already_extracted(tmp_path: Path) -> None:
    item = _item(tmp_path)
    _save_both(tmp_path, item)
    fake = _Fake()
    report = _run(tmp_path, fake, [(item, "drive1")])
    assert fake.downloads == [] and report.already_done == [ID]


def test_fetch_stops_before_downloading_when_disk_is_low(tmp_path: Path) -> None:
    fake = _Fake()
    report = _run(tmp_path, fake, [(_item(tmp_path), "drive1")], free=MARGIN_BYTES - 1)
    assert fake.downloads == []
    assert report.stopped is not None and "free" in report.stopped


def test_fetch_records_a_failed_download_and_carries_on(tmp_path: Path) -> None:
    fake = _Fake(fail_download="drive1")
    report = _run(tmp_path, fake, [(_item(tmp_path), "drive1"), (_item(tmp_path, ID2, 2), "d2")])
    assert report.failed == [(ID, "download: not a zip (quota page?)")]
    assert report.done == [ID2]


def test_check_download_calls_a_quota_page_a_refusal(tmp_path: Path) -> None:
    page = tmp_path / "c.zip"
    page.write_bytes(b"<!DOCTYPE html><html>Google Drive - Quota exceeded</html>")
    with pytest.raises(Refused):
        check_download(page, None)


class _Refusing(_Fake):
    """Drive answers the listed downloads with a refusal page (seen 2026-10-20 from 20:51)."""

    def __init__(self, refuse: set[str]) -> None:
        super().__init__()
        self.refuse = refuse

    def download(self, drive_id: str, target: Path) -> None:
        if drive_id in self.refuse:
            self.downloads.append(drive_id)
            raise Refused("not a zip (quota page?)")
        super().download(drive_id, target)


def _clips(tmp_path: Path, n: int) -> list[tuple[VideoItem, str]]:
    return [(_item(tmp_path, f"Subject1/Activity{i}/Trial1/Subject1Activity{i}Trial1Camera1"),
             f"d{i}") for i in range(n)]


def test_fetch_stops_after_refusals_in_a_row(tmp_path: Path) -> None:
    fake = _Refusing({f"d{i}" for i in range(6)})
    report = _run(tmp_path, fake, _clips(tmp_path, 6))
    assert fake.downloads == ["d0", "d1", "d2"]
    assert len(report.failed) == REFUSALS_TO_STOP == 3
    assert report.stopped is not None and "refused" in report.stopped


def test_a_good_download_resets_the_refusal_count(tmp_path: Path) -> None:
    fake = _Refusing({"d0", "d1", "d3", "d4"})
    report = _run(tmp_path, fake, _clips(tmp_path, 6))
    assert report.stopped is None
    assert len(report.done) == 2 and len(report.failed) == 4


def test_other_download_errors_do_not_stop_the_run(tmp_path: Path) -> None:
    fake = _Fake(fail_download="d0")
    clips = _clips(tmp_path, 1) * 4
    report = _run(tmp_path, fake, clips)
    assert report.stopped is None and len(report.failed) == 4


def test_fetch_keeps_the_zip_when_extraction_fails(tmp_path: Path) -> None:
    item = _item(tmp_path)
    report = _run(tmp_path, _Fake(fail_extract=True), [(item, "drive1")])
    assert report.failed == [(ID, "lite: 1 frames extracted, 2 in the source")]
    assert item.source.exists()


def test_fetch_extracts_a_zip_left_by_an_interrupted_run(tmp_path: Path) -> None:
    item = _item(tmp_path)
    item.source.parent.mkdir(parents=True)
    _zip(item.source)
    fake = _Fake()
    report = _run(tmp_path, fake, [(item, "drive1")])
    assert fake.downloads == [] and report.done == [ID]


def test_fetch_without_a_downloader_lists_missing_zips(tmp_path: Path) -> None:
    present = _item(tmp_path)
    present.source.parent.mkdir(parents=True)
    _zip(present.source)
    report = fetch([(present, "drive1"), (_item(tmp_path, ID2, 2), "d2")], tmp_path,
                   download=None, process=_Fake().process, free_bytes=lambda: 0, pause_s=0.0)
    assert report.done == [ID] and report.missing == [ID2]
    assert report.failed == [] and report.stopped is None


def test_fetch_checks_a_zip_it_did_not_download(tmp_path: Path) -> None:
    item = _item(tmp_path)
    item.source.parent.mkdir(parents=True)
    item.source.write_bytes(b"PK cut short by the browser")
    report = _run(tmp_path, _Fake(), [(item, "drive1")])
    assert report.done == [] and report.failed[0][0] == ID
    assert report.failed[0][1].startswith("zip: ") and item.source.exists()


def test_stray_files_lists_what_is_not_an_expected_zip(tmp_path: Path) -> None:
    for name in ("Subject1Activity1Trial1Camera1.zip", "Subject1Activity1Trial1Camera1 (1).zip",
                 "Subject1Activity1Trial1Camera2.zip.crdownload", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    assert stray_files(tmp_path, LINKS) == [
        "Subject1Activity1Trial1Camera1 (1).zip",
        "Subject1Activity1Trial1Camera2.zip.crdownload",
        "notes.txt",
    ]


def test_stray_files_of_a_missing_folder_is_empty(tmp_path: Path) -> None:
    assert stray_files(tmp_path / "none", LINKS) == []


def test_plan_clips_for_chosen_subjects(tmp_path: Path) -> None:
    clips = plan_clips(LINKS, _omnifall(tmp_path), tmp_path / "raw", subjects=[1])
    assert [(item.video_id, item.cam, drive) for item, drive in clips] == [
        (ID, 1, "drive1"), (ID2, 2, "drive2"),
    ]
    assert clips[0][0].source == tmp_path / "raw" / "Subject1Activity1Trial1Camera1.zip"


def test_plan_clips_needs_a_link_for_every_omnifall_clip(tmp_path: Path) -> None:
    links = {ID: "drive1"}
    with pytest.raises(ValueError, match="Camera2"):
        plan_clips(links, _omnifall(tmp_path), tmp_path / "raw", subjects=[1])


def test_main_with_everything_done_needs_no_network(tmp_path: Path) -> None:
    _omnifall(tmp_path)
    page = tmp_path / "page.html"
    page.write_text("".join(
        f'<h5>Subject1Activity1Trial1</h5><a href="https://x/uc?id={d}&export=download">'
        f"Camera{c}</a>" for c, d in ((1, "drive1"), (2, "drive2"))
    ))
    for item in (_item(tmp_path), _item(tmp_path, ID2, 2)):
        _save_both(tmp_path, item)
    assert main([
        "--data-root", str(tmp_path), "--page", str(page), "--subjects", "1",
        "--models-dir", str(tmp_path), "--free-space-path", str(tmp_path),
    ]) == 0


def _page(tmp_path: Path) -> Path:
    page = tmp_path / "page.html"
    page.write_text("".join(
        f'<h5>Subject1Activity1Trial1</h5><a href="https://x/uc?id={d}&export=download">'
        f"Camera{c}</a>" for c, d in ((1, "drive1"), (2, "drive2"))
    ))
    return page


def test_main_imports_from_a_chosen_folder_without_downloading(tmp_path: Path) -> None:
    _omnifall(tmp_path)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _zip(inbox / "Subject1Activity1Trial1Camera1.zip")
    for item in (_item(tmp_path), _item(tmp_path, ID2, 2)):
        _save_both(tmp_path, item)
    assert main([
        "--data-root", str(tmp_path), "--page", str(_page(tmp_path)), "--subjects", "1",
        "--models-dir", str(tmp_path), "--raw-dir", str(inbox), "--no-download",
    ]) == 0


def test_main_fails_on_a_stray_file_in_the_folder(tmp_path: Path) -> None:
    _omnifall(tmp_path)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "Subject1Activity1Trial1Camera1 (1).zip").write_bytes(b"")
    for item in (_item(tmp_path), _item(tmp_path, ID2, 2)):
        _save_both(tmp_path, item)
    assert main([
        "--data-root", str(tmp_path), "--page", str(_page(tmp_path)), "--subjects", "1",
        "--models-dir", str(tmp_path), "--raw-dir", str(inbox), "--no-download",
    ]) == 1
