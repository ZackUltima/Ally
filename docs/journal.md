# Ally — engineering journal

Five lines per week, written on Friday. Newest entry first.
Format: **Done · Measured · Blocked · Decided · Next.** Numbers here feed the final report's implementation chapter.

---

## Week 0 — 21–27 Sep 2026
- Done: plan v1.1; `CLAUDE.md`; `docs/context/` split; Sprint 0 scaffold — package tree per §7.2, `contracts.py` + tests, `config.py` (documented numbers only), Makefile + `tasks.ps1`, requirements tiers, pre-commit block hook, CI workflow, replay-harness test, `benchmark_fps.py`, `extract_keypoints.py`; uv-managed Python 3.11 venv; `test`/`lint` green locally.
- Measured: — (FPS/VRAM needs the CUDA install; not run yet).
- Blocked: full runtime (`install`) and the pose benchmark not run; CI never executed; GNU make absent (`tasks.ps1` mirror verified instead).
- Decided: ADR-001…004; Sprint 0 tooling notes in `context/decisions.md` (uv, `--universal` lock, `EventType.SUPPRESSED`).
- Next: `install` → `bench` FPS/VRAM row; SPTA supervisor shortlist; download UR Fall; commit the scaffold on `develop-1`.
