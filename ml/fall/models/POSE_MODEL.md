# Pose models (D-13)

The `.task` files are not committed (D-52); they are downloaded to `~/ally-data/models/` from Google's versioned URLs below, never from `latest`. The phone must run the same file (D-13): the Android build will use the lite file with the same checksum.

| Model | URL | Size (bytes) | SHA-256 | Uploaded |
|---|---|---|---|---|
| Pose Landmarker lite, float16, version 1 (default, D-13 stated condition) | https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task | 5,777,746 | `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a` | 2023-04-27 |
| Pose Landmarker full, float16, version 1 (also saved, PO 2026-10-19, in case Spike 2 picks it) | https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task | 9,398,198 | `5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1` | 2023-04-28 |

Python package used for extraction: `mediapipe` at the version pinned in `ml/fall/requirements.txt`.

Downloaded and checked 2026-10-19 (sizes match Google's bucket listing).
