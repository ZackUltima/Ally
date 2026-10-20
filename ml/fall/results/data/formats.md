# Fall datasets: raw formats (checked 2026-10-19, before Task 6)

Sources: each dataset's official page (Official), OmniFall's label files and STRUCTURE.md (Preprint dataset card), opened 2026-10-19. Licences are re-read when each dataset is downloaded (D-19).

| Dataset | Host and download unit | Raw format | Frame times | Licence | OmniFall `path` example (cams) |
|---|---|---|---|---|---|
| GMDCSA-24 | Zenodo record 12921216: one zip, 1,107.5 MB, v2.0 | Videos (format checked on unzip), folders `Subject 1`–`Subject 4` | From the video decoder | CC BY 4.0 (Zenodo); the GitHub mirror says MIT | `Subject_1/ADL/01` (cam 1 only) |
| CAUCAFall | Mendeley Data 7w7fccy7ky v5 (2025-03-18; v4 was checked on 2026-10-19); only the 100 `.avi` files are downloaded (about 130 MB of 8.3 GB), each checked against Mendeley's SHA-256 | Per action: an `.avi` video, the `.png` frames and per-frame `.txt` labels. **Measured 2026-10-20:** the `.avi` files are 720 × 480 at 20 fps (the paper's 1080 × 960, 23 fps describes the camera [31]); the 100 `.avi` files are byte-identical in v4 and v5 | From the `.avi` (steady 50 ms steps; OmniFall's last label ends within 0.07 s of each video's end) | CC BY 4.0 (Mendeley record, read 2026-10-20) | `adl/HopS1` (cam 1 only) |
| Le2i | Original host offline; mirror and licence to be confirmed before download (D-54 risk 3) | `.avi` videos (to confirm) | From the video decoder | Unknown until checked | `Coffee_room_01/video_14` (cam 1–6 = scenes) |
| UP-Fall | HAR-UP Google Site: one zip **per subject, activity, trial and camera** | **Image frames** (zip of images), 2 cameras | From the image names or the trial's CSV (to confirm on the first zip) | No licence stated; "publicly available", cite the Sensors 2019 paper (D-54 risk 4) | `Subject1/Activity1/Trial1/Subject1Activity1Trial1Camera1` (cams 1, 2) |

## Consequences for the plan

- **Task 6 input is a frame source**, not only a video file: UP-Fall ships frames, so `folder_frames` is needed (ruling in the ledger).
- **Disk:** UP-Fall's download unit is one trial-camera zip, far smaller than one subject (about 50 GB), so the 15 GB free-space margin can be kept without special measures.
- **Labels in the staged data** (from OmniFall's label files): only ids 0–9 occur. `lie_down` and `lying` are rare (Le2i 1 and 0 segments, CAUCAFall 1 and 1, UP-Fall 6 and 118). This matters for training in F2 (class weights, D-55 item 3) and for the fallen-versus-lying question (D-54 risk 10).
