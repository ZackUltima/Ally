# Ally — engineering journal

Five lines per week, written on Friday. Newest entry first.
Format: **Done · Measured · Blocked · Decided · Next.** Numbers here feed the final report's implementation chapter.

---

## Week 0 — 21–27 Sep 2026
- Done: plan v1.1; `CLAUDE.md`; `docs/context/` split; Sprint 0 scaffold — package tree per §7.2, `contracts.py` + tests, `config.py` (documented numbers only), Makefile + `tasks.ps1`, requirements tiers, pre-commit block hook, CI workflow, replay-harness test, `benchmark_fps.py`, `extract_keypoints.py`; uv-managed Python 3.11 venv; scaffold committed on `develop-1`. Then (22 Sep): `fall_features.py` (§6.1 step 2 features + window stats + online extractor, 30 tests), `RuleFallDetector` (step 3a, 20 tests), `prepare_urfall.py` (download/layout/labels; verified end-to-end on `fall-01`: 160 frames, 160 label rows), `extract_keypoints.py` reads PNG sequences; 82 tests + lint green.
- Measured: — (FPS/VRAM needs the CUDA install; not run yet).
- Blocked: full runtime (`install`) and the pose benchmark not run; CI never executed; GNU make absent (`tasks.ps1` mirror verified instead); UR Fall subject map not published → `subjects.csv` must be filled by hand before LOSO-CV.
- Decided: ADR-001…004; Sprint 0 tooling notes in `context/decisions.md` (uv, `--universal` lock, `EventType.SUPPRESSED`); Week 0 perception notes (feature conventions, `v_thr`/`hip_height_max` left required-not-defaulted, subject map never guessed).
- Next: `install` → `bench` FPS/VRAM row; SPTA supervisor shortlist; full UR Fall pull (`prepare_urfall.py`, ~4 GB) → `extract_keypoints.py` → rule baseline numbers on real keypoints; fill `subjects.csv`.
