# CAUCAFall: keypoints and windows (F1 Task 8, built 2026-10-20)

Our measurements from the build run (`python -m ally_fall.build --datasets GMDCSA24 caucafall`, split by dataset from the saved arrays), unless a source is named.

## Source

- **Dataset:** CAUCAFall [31], Mendeley Data `7w7fccy7ky` version 5 (published 2025-03-18; DOI 10.17632/7w7fccy7ky.5). Downloaded 2026-10-20.
- **What was downloaded:** only the 100 `.avi` videos (about 130 MB of the record's 8.3 GB; the rest is the same footage as `.png` frames plus per-frame `.txt` boxes). Each file was fetched from its Mendeley `download_url` (public API, `/public-api/datasets/7w7fccy7ky`) and checked against the SHA-256 Mendeley lists: 100 of 100 match. The 100 videos are byte-identical in v4 and v5 (same SHA-256), so the version change does not affect OmniFall's labels.
- **Licence:** CC BY 4.0, read on the Mendeley record on 2026-10-20 (credit, link to the licence, say what was changed; our change is keypoint extraction). Recorded in D-19 (PO, 2026-10-20). The videos stay in `~/ally-data/raw/` and never go into the repository (D-52).
- **Labels and split:** OmniFall [28], Hugging Face commit `83572a37b9e3081df8c06a56874b1d1f2a19386c` (`omnifall.REVISION`); cross-subject split: train subjects 1–7, validation 10, test 8 and 9 (D-54), matching OmniFall's own split files (`test_split_matches_omnifall`). All 100 downloaded videos map to OmniFall's 100 label paths, none missing or extra.

## Extraction

| | Value |
|---|---|
| Videos | 100 (10 people × 5 falls + 5 daily activities), one camera |
| Frame size and rate | 720 × 480 at 20 fps in all 100 (decoder times in steady 50 ms steps; frame counts match). The paper's 1080 × 960, 23 fps [31] describes the camera, not the released files. GMDCSA-24 is 1280 × 720, so the two differ in shape (3:2 against 16:9); keypoints are saved normalised per axis, and F2 decides how to handle this |
| Label timing | OmniFall's last label ends within 0.07 s of each video's end, so the labels follow these 20 fps files |
| Pose models | MediaPipe Pose Landmarker lite and full, v1 (`models/POSE_MODEL.md`) |
| Files written | 200 (100 per model), 0 failed |
| Time | 770 s (12.8 min) for both models, on the laptop CPU in WSL |
| Keypoint size | 16 MB for both models (8.1 MB each) |
| Videos deleted | None (deletion stays the PO's call before UP-Fall) |

## Windows per split (lite; full is identical in videos, ticks, windows and labels)

A window is 30 ticks (3 s at 10 fps) labelled by its last tick (D-55 items 1–2); windows ending in a label gap are not made.

| Split | Subjects | Videos | Ticks | Windows | Fall class (fall ∪ fallen) | Ticks with no pose, lite / full |
|---|---|---|---|---|---|---|
| Train | 1–7 | 69 | 6,536 | 4,534 | 2,038 (44.9%) | 12.0% / 11.0% |
| Validation | 10 | 10 | 1,240 | 950 | 465 (48.9%) | 12.6% / 11.5% |
| Test | 8, 9 | 20 | 2,125 | 1,545 | 717 (46.4%) | 24.7% / 19.1% |
| **Total** | 10 | **99** | 9,901 | **7,029** | 3,220 | |

## Label counts (windows)

| Label | Train | Validation | Test |
|---|---|---|---|
| walk | 703 | 83 | 167 |
| fall | 487 | 60 | 128 |
| fallen | 1,551 | 405 | 589 |
| sit_down | 166 | 20 | 40 |
| sitting | 288 | 47 | 122 |
| lie_down | 43 | 0 | 0 |
| lying | 25 | 0 | 0 |
| stand_up | 189 | 35 | 51 |
| standing | 54 | 16 | 0 |
| other | 1,028 | 284 | 448 |
| kneel_down, kneeling, squat_down, squatting, crawl, jump | 0 | 0 | 0 |

OmniFall labels CAUCAFall's hop and kneel videos as *other* (for example `adl/KneelS1`, label 9, 0–10.95 s), so kneeling adds no kneel windows.

**Training counts so far (GMDCSA-24 + CAUCAFall)**, saved as `built/<pose>/label_counts_train.json` (D-55 item 3; the same for lite and full): walk 894, fall 801, fallen 2,408, sit_down 243, sitting 990, lie_down 55, lying 166, stand_up 275, standing 198, other 1,736; the last six labels 0. Total 7,766 training windows.

## Left out

| Video | Reason |
|---|---|
| `forward/FallForwardS2` | Overlapping labels (0.20 s): *fall* and *fallen* share 0.2 s, over the one-tick limit (PO 2026-10-20, option A). Both labels are in the fall class. 19 other CAUCAFall videos have overlaps of 0.1 s or less (27 boundaries), trimmed as rounding |

## Where the pose is lost (lite / full, share of ticks with no pose)

| Person | Split | No pose |
|---|---|---|
| 1, 2, 4, 7 | Train | 2.6–8.7% |
| 3 | Train | 19.6% / 22.4% |
| 5 | Train | 13.4% / 11.9% |
| 6 | Train | 25.0% / 18.8% |
| 9 | Test | 6.8% / 8.6% |
| 10 | Validation | 11.3% / 10.5% |
| **8** | **Test** | **44.0% / 33.0%** |

Worst videos: `forward/FallForwardS8` 100% / 100% (no pose in any frame), `side/FallRightS8` 86% / 83%, `side/FallLeftS6` 74% / 79%. The two videos viewed (`FallForwardS8`, `FallLeftS6`; frames saved outside the repository) are dim night footage from a high ceiling corner, with the person lying flat on the floor; the person is clearly visible to a human. CAUCAFall records natural, artificial and night light on purpose [31].

## Notes for the PO

1. **Test person 8 loses the pose in a third to almost half of ticks,** mainly in dim, top-down fall videos. Those test windows are real misses for a pose-based model, so they lower the CAUCAFall test score. They are kept: the build does not drop videos for having no pose, and D-55's missing-point handling is what the model sees on the phone too.
2. **This view is not Ally's.** Ally's phone sits at table height in a lit room (a dark room is a stated limitation, design plan part 1); CAUCAFall's camera is in an upper wall corner at 2.15 m [31] (D-54 risk 2). **Decided (PO, 2026-10-20; D-54):** F3 reports the test score with and without CAUCAFall's dim test videos, so the headline does not hide this; the dim list is fixed by brightness before any model is trained.

**Report approved by the PO on 2026-10-20** ("approve for now, open to changes in the future").
3. **Full keeps the person more often than lite here** (test 19.1% against 24.7% no pose), unlike GMDCSA-24 where the two were level. No pose does not measure whether the points are right, so the lite-versus-full choice stays with Spike 2 (D-13).
4. **Kneel, squat, crawl and jump still have no training windows** after two datasets; lie_down has 55 and lying 166. Class weights are set after all four datasets (D-55 item 3).
