# Ally — CLAUDE.md

Ally is a CAT405 final-year project (USM, 2026/27): a laptop + webcam system that detects falls,
prolonged inactivity and low mood, **talks to the person first**, and escalates to a guardian on
Telegram only after verification. Raw video never leaves the device.

Full context lives in `docs/context/` — read only the file(s) relevant to the task (see `docs/README.md`).
The graded source document is `Ally_Project_Plan.html` (v1.1); the context docs are derived from it.

## Stack
- Python 3.11 in a venv (3.13 is installed on the machine — do not use it; CV/ML wheels lag).
- Vision: OpenCV, Ultralytics YOLO-pose nano **or** MediaPipe Pose Landmarker (decided in Sprint 1), ONNX Runtime (CUDA EP).
- ML: PyTorch (CUDA), scikit-learn, LightGBM. GPU is an RTX 2050 with 4 GB VRAM — small models only.
- Speech: faster-whisper on **CPU int8** (GPU is reserved for vision), Piper TTS, sounddevice, Silero/WebRTC VAD.
- LLM/VLM: Anthropic Python SDK behind `ally/agent/llm_client.py` (`LLMClient` interface — nothing else imports `anthropic`).
- Backend: FastAPI + Uvicorn, SQLModel on SQLite, Jinja2 dashboard. Notifications: python-telegram-bot.
- Tooling: pytest, ruff, pre-commit, Makefile. Pinned `requirements.txt`.

## Layout (see `docs/context/architecture.md` for the full tree)
```
ally/perception  camera.py pose.py fall_features.py fall_model.py face_emotion.py presence.py
ally/fusion      state.py event_engine.py (pure FSM, no I/O) cooldowns.py
ally/agent       dialogue.py tools.py llm_client.py prompts/persona.md offline_fallback.py
ally/speech      vad.py stt.py tts.py
ally/notify      telegram_bot.py        ally/api  server.py dashboard/
ally/storage     db.py models.py retention.py
ally/contracts.py  pydantic models exchanged between modules
ally/main.py       starts workers over bounded queues; --replay <session> replaces the webcam
scripts/  record_session.py extract_keypoints.py train_fall.py eval_fall.py train_fer.py run_scenarios.py benchmark_fps.py
tests/    unit/ (FSM, features, tools)  replay/ (recorded sessions → expected events)  scenarios/ (30 dialogue cases)
docs/     context/  journal.md  proposal/  srd/  final_report/
```

## Module contracts (typed, in `ally/contracts.py`)
- perception → fusion: `KeypointFrame` (timestamp, keypoints[N,3], bbox, face_valence | None, person_present).
- fusion → agent: `PersonState` (posture, motion, mood, presence) and `Event` (type, confidence, keyframe_id, state).
- agent → fusion: `EscalationRequest` / `StandDownRequest` (reason, urgency). **Requests, not actions.**
- fusion → notify: `Notification` (event, keyframe, transcript). Only `event_engine` may create one.
Each module is testable at its boundary with fixtures; never reach across a boundary to a concrete class.

## Hard rules
1. **The FSM owns escalation.** The LLM classifies a check-in reply as `fine` / `help` / `unclear` and may *request*
   escalation or stand-down; `ally/fusion/event_engine.py` decides. `unclear` or silence after 2 prompts → `CONFIRMED`.
   The model has no tool that sends a Telegram message. (ADR-001)
2. **Never write raw frames to disk.** Frames live in the RAM ring buffer; only one JPEG keyframe per event is
   persisted (7-day TTL). `record_session.py` records keypoints, audio and events — not video.
3. **Privacy boundary.** Only conversation text, a JSON state summary and one keyframe per event leave the device.
   Respect `ALLY_TEXT_ONLY_VERIFY=true` (no image upload).
4. **Secrets.** Keys in `.env` only; `.env.example` documents them. Never print or log a key.
5. **Numbers are owned by the human.** Do not change a threshold, target (recall, FA/h, latency) or cooldown without
   adding a `SYNTHESIS` note to `docs/context/decisions.md`. Do not invent citations or statistics.
6. **No medical or emergency-device claims** in prompts, UI text or docs. Ally is a research prototype.
7. **Telegram bot** rejects any chat ID other than `TELEGRAM_GUARDIAN_CHAT_ID`. Dashboard binds to LAN only.
8. Capture resolution is 640×480. Queues between workers are bounded, drop-oldest. Perception must stay ≥ 15 FPS.
9. Retention job must be tested to actually delete. Keyframes are deletable from the dashboard.

## Commands (Makefile — create in Sprint 0; targets below are the contract)
- `make test`      pytest `tests/unit` + `tests/replay`
- `make lint`      ruff check + ruff format --check
- `make replay SESSION=<name>`   run a recorded session through perception → fusion → agent, print events
- `make scenarios` run the 30 scripted dialogue scenarios through the agent harness (needs API key)
- `make run`       live app;  `make bench`  FPS/VRAM profile
- Experiments: `make train-fall`, `make eval-fall`, `make train-fer`, `make eval-fer` (fixed seeds, results to `results/`)

## Definition of done for any task
- `make test` and `make lint` green.
- Touched `perception/` or `fusion/` → `make replay` on every session in `tests/replay/` still yields expected events.
- Touched `agent/` → `make scenarios` escalation-decision accuracy unchanged or better; injection cases still pass.
- New decision or number → paragraph in `docs/context/decisions.md`. Five lines in `docs/journal.md`.
- Report what was verified and how; if something was not run, say so.

## Working with Claude Code on this repo
- One session per task; small vertical slices ("FSM `VERIFY` transitions + tests", not "build the event system").
- Use plan mode before touching the FSM, thresholds, the privacy boundary, or the escalation path; propose 2 options.
- Point the session at the one `docs/context/*.md` it needs; do not paste the whole plan.
- Before Sprint 6 demo hardening, run an adversarial pass in a fresh session: "find what's wrong, especially
  the escalation path, retention and the Telegram handler" — then `/security-review`.
- When a session runs long, ask it for a handoff prompt for the next session.
