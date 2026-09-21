# Replay sessions

One directory per recorded session (ADR-004), produced by `scripts/record_session.py`:

```
tests/replay/<session>/
├── keypoints.npz          # ts[T], keypoints[T, N, 3], bbox[T, 4], person_present[T]  — never frames
├── audio.wav              # optional, mic only
├── meta.json              # pose backend, fps, capture size, recorded_at, notes
└── expected_events.json   # ordered [{"type": "SUSPECTED_FALL", "t_min": 12.0, "t_max": 14.0}, ...]
```

`test_replay_sessions.py` runs every session through perception → fusion → agent (offline fallback, no API
key) and asserts the engine emits exactly `expected_events.json` within tolerance. Sessions are added from
Sprint 1. Keep them short (< 2 min) and small (< 2 MB); longer ones live in `data/sessions/` (git-ignored).
