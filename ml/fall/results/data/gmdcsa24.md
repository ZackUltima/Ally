# GMDCSA-24: keypoints and windows (F1 Task 8, built 2026-10-20)

Our measurements from the build run (`python -m ally_fall.build`, via `build()`), unless a source is named.

## Source

- **Dataset:** GMDCSA-24 [27], Zenodo record 12921216, v2.0, one zip of 1,107,543,412 bytes; MD5 `49bf4eb15a84cc84cb0a4f9c6ddd59e6`, the same as Zenodo's. Downloaded 2026-10-19.
- **Licence:** CC BY 4.0 (D-54). The zip and the videos stay in `~/ally-data/raw/` and never go into the repository (D-52).
- **Labels and split:** OmniFall [28], Hugging Face commit `83572a37b9e3081df8c06a56874b1d1f2a19386c` (pinned in `omnifall.REVISION`); cross-subject split: train subjects 1 and 3, validation 2, test 4 (D-54). This matches OmniFall's own GMDCSA-24 split files exactly (`test_split_matches_omnifall`).

## Extraction

| | Value |
|---|---|
| Videos | 160 (79 fall, 81 ADL); 1280 × 720 in the one video checked |
| Pose models | MediaPipe Pose Landmarker lite and full, v1 (`models/POSE_MODEL.md`) |
| Files written | 320 (160 per model), 0 failed |
| Time | 1,355 s (22.6 min) for both models, on the laptop CPU in WSL |
| Keypoint size | 28 MB for both models (the videos are about 1.1 GB) |
| Videos deleted | None. Deleting after verification was blocked by the auto-mode safety check on 2026-10-20; the videos are kept, and the PO decides before UP-Fall |

Real-person check (Task 6 Step 6): Subject 1 fall videos 01–04 drawn on frames; the points sit on the person (PO, "points OK", 2026-10-20).

## Windows per split (lite; full is identical in videos, ticks, windows and labels)

A window is 30 ticks (3 s at 10 fps) labelled by its last tick (D-55 items 1–2); windows ending in a label gap are not made.

| Split | Subjects | Videos | Ticks | Windows | Fall class (fall ∪ fallen) | Ticks with no pose, lite / full |
|---|---|---|---|---|---|---|
| Train | 1, 3 | 74 | 5,386 | 3,232 | 1,171 (36.2%) | 3.9% / 4.1% |
| Validation | 2 | 48 | 4,546 | 3,141 | 1,053 (33.5%) | 3.9% / 4.1% |
| Test | 4 | 37 | 2,894 | 1,812 | 454 (25.1%) | 4.6% / 4.8% |
| **Total** | 4 | **159** | 12,826 | **8,185** | 2,678 | |

## Label counts (windows)

| Label | Train | Validation | Test |
|---|---|---|---|
| walk | 191 | 221 | 92 |
| fall | 314 | 375 | 46 |
| fallen | 857 | 678 | 408 |
| sit_down | 77 | 162 | 4 |
| sitting | 702 | 299 | 491 |
| lie_down | 12 | 100 | 78 |
| lying | 141 | 89 | 88 |
| stand_up | 86 | 53 | 10 |
| standing | 144 | 246 | 10 |
| other | 708 | 918 | 585 |
| kneel_down, kneeling, squat_down, squatting, crawl, jump | 0 | 0 | 0 |

Training counts are also saved as `built/<pose>/label_counts_train.json` (D-55 item 3).

## Left out

| Video | Reason |
|---|---|
| `Subject_3/ADL/09` | Overlapping labels (5.28 s): OmniFall gives stand_up 0–7.775 s across sitting 2.5–6.0 s and standing 6.2–7.0 s. Overlaps of 0.1 s or less (one tick) are rounding and are trimmed (PO 2026-10-20, option A); in GMDCSA-24, 32 such boundaries in 23 videos. |

## Notes for the PO

1. **Seven labels have more than 100 training windows in this dataset;** lie_down has 12 and the last six labels have none. One dataset of four people is not the whole picture: the class weights (D-55 item 3) are set after all four datasets are built.
2. **The test person has only 46 fall windows** (one person, 37 videos), so this dataset alone cannot give a stable test score. That is expected: the headline uses all 9 test people across four datasets (D-54).
3. **Lite and full lose the person about equally often** (about 4% of ticks), and full slightly more. The 4-video check on 2026-10-20 suggested full holds on longer when a person lies at the frame edge; over all 160 videos that does not show as fewer missing ticks. "No pose" does not measure whether the points are right, so the lite-versus-full choice stays with Spike 2 (D-13).
