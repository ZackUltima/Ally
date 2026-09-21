# Architecture

Source: `Ally_Project_Plan.html` v1.1 §5 (pipeline, FSM, data flow) and §7.2 (repository layout).
Decisions that changed this section: ADR-001 (FSM owns escalation), ADR-004 (replay harness) — `decisions.md`.

## Shape

Ally is a modular, event-driven pipeline. Perception runs locally on the laptop GPU; only compact text and
one event keyframe cross the privacy boundary to the cloud model and the guardian channel.

```
LOCAL DEVICE — raw video never leaves this boundary
  USB webcam ──► capture (OpenCV, 640×480, ≥15 FPS)
     │  KeypointFrame
     ▼
  perception (GPU, ONNX/PyTorch)
     pose estimation (YOLO-pose nano | MediaPipe) → fall/inactivity temporal classifier
     face detect → FER CNN → mood EMA
     │  KeypointFrame (keypoints, bbox, face_valence, person_present)
     ▼
  fusion — state + event engine (pure FSM, cooldowns)          ◄── EscalationRequest / StandDownRequest ──┐
     NORMAL → SUSPECTED_* → VERIFY → CONFIRMED | STOOD_DOWN | FALSE_ALARM ; COMPANION_CHAT                  │
     │  Event + PersonState                                                                                 │
     ▼                                                                                                     │
  agent — dialogue orchestrator (persona prompt, short-term memory, tools) ─────────────────────────────────┘
     speech I/O: mic → VAD → STT (faster-whisper, CPU int8) ; TTS (Piper) → speaker
     offline fallback: scripted check-in + keyword spotting + local alarm
     │  Notification (created ONLY by the event engine)
     ▼
  notify — Telegram bot (alert + keyframe + transcript; guardian replies "ok")
  storage — SQLite (events, transcripts, mood samples, keyframes 7-day TTL) ; FastAPI dashboard (LAN only)

CLOUD (text + 1 keyframe per event only): LLM dialogue with tool use · VLM keyframe verification
```

## Module contracts (`ally/contracts.py`, pydantic)

| Boundary | Type | Fields (minimum) |
|---|---|---|
| perception → fusion | `KeypointFrame` | `ts`, `keypoints[N,3]` (x, y, conf), `bbox`, `face_valence: float | None`, `person_present: bool` |
| fusion → agent | `PersonState` | `posture`, `motion_energy`, `mood`, `presence`, `inactive_for_s` |
| fusion → agent | `Event` | `type` (SUSPECTED_FALL, SUSPECTED_INACTIVE, VERIFY, CONFIRMED, …), `confidence`, `keyframe_id`, `state: PersonState` |
| agent → fusion | `EscalationRequest`, `StandDownRequest` | `reason`, `urgency` — **requests; the engine decides** |
| agent → fusion | `ReplyLabel` | `label: fine | help | unclear`, `transcript` — structured output from the model |
| fusion → notify | `Notification` | `event`, `keyframe_path`, `transcript`, `blur_face: bool` |

Rules: modules import only `contracts` and their own package; each is testable at its boundary with fixtures;
the FSM class has no I/O (no camera, no network, no clock — time is passed in).

## Runtime model

- `main.py` starts workers connected by **bounded queues (drop-oldest)**: capture → perception → fusion → agent.
  Perception must hold ≥ 15 FPS; a slow consumer drops frames rather than back-pressuring capture.
- Frames live in a **RAM ring buffer (last 10 s)** so a keyframe from just before the event can be selected;
  never written to disk. At 640×480 ≈ 140 MB; 1080p would be ≈ 930 MB — resolution is fixed at 480p.
- **GPU placement**: pose + FER on the RTX 2050; faster-whisper (int8) and Piper on CPU. A per-model VRAM/CPU
  table measured in Sprint 1 is the NFR-4 budget of record (< 3 GB total).
- **`--replay <session>`** replaces the webcam with a recording from `scripts/record_session.py`
  (keypoints + audio + event log; never raw frames). Same code path from perception onward (ADR-004).

## Event engine (§5.1, corrected per ADR-001)

| State | Entry | Action | Exit |
|---|---|---|---|
| `NORMAL` | default | run perception; sample mood every 5 s; update inactivity timer | fall classifier ≥ τ_fall → `SUSPECTED_FALL`; inactivity ≥ T (default 30 min, person present) → `SUSPECTED_INACTIVE`; mood negative > N min and cooldown elapsed → `COMPANION_CHAT` |
| `SUSPECTED_*` | trigger fired | freeze keyframe; 3 s debounce (posture stays "on floor" / no motion) | debounce passes → `VERIFY`; else → `NORMAL` (log as suppressed) |
| `VERIFY` | debounce passed | in parallel: (a) VLM keyframe query (structured JSON answer); (b) canned TTS "…are you all right? Please say something."; listen 20 s, up to 2 prompts; model labels each reply `fine` / `help` / `unclear` | label `fine` **and** VLM does not confirm fall → `STOOD_DOWN`. Label `help` or `unclear`, no reply after 2 prompts, or VLM + posture agree on fall → `CONFIRMED`. **Computed by the FSM; the model never decides.** |
| `CONFIRMED` | verification positive / unanswered | engine creates `Notification` (type, time, keyframe, transcript); keep talking to reassure; retry until guardian acks; if offline → local audible alarm + queued retry | guardian acks or 10 min → `NORMAL` with cooldown |
| `STOOD_DOWN` / `FALSE_ALARM` | person OK / VLM negative | log outcome (used to tune τ_fall); optional light chat | → `NORMAL`, 5 min cooldown |
| `COMPANION_CHAT` | negative mood or scheduled check-in | LLM conversation (open question, active listening, suggest activity); ends politely on silence | → `NORMAL`; log mood before/after |

Every transition in this table has a unit test in `tests/unit/test_event_engine.py`.

## Privacy boundary (§5.2)

Leaves the device: (1) JSON state summary + conversation text to the LLM API; (2) one JPEG keyframe per event
to the VLM and to Telegram (face-blur on by default for Telegram); (3) nothing else. `ALLY_TEXT_ONLY_VERIFY=true`
disables image upload entirely. Dashboard binds to the LAN interface with authentication (token or HTTPS; not
basic auth over plain HTTP). Telegram handler rejects every chat ID except the guardian's.

## Repository layout (§7.2)

```
ally/
├── ally/
│   ├── perception/   camera.py · pose.py · fall_features.py · fall_model.py · face_emotion.py · presence.py
│   ├── fusion/       state.py · event_engine.py (FSM) · cooldowns.py
│   ├── agent/        dialogue.py · tools.py · llm_client.py · prompts/persona.md · offline_fallback.py
│   ├── speech/       vad.py · stt.py · tts.py
│   ├── notify/       telegram_bot.py
│   ├── api/          server.py · dashboard/ (templates, static)
│   ├── storage/      db.py · models.py · retention.py
│   ├── contracts.py  pydantic models at module boundaries
│   ├── config.py     pydantic settings; .env for keys
│   └── main.py       workers over bounded queues; --replay <session>
├── scripts/          prepare_urfall.py · extract_keypoints.py · train_fall.py · eval_fall.py
│                     train_fer.py · eval_fer.py · run_scenarios.py · benchmark_fps.py · record_session.py
├── notebooks/        01_pose_benchmark · 02_fall_features · 03_fer
├── data/             (git-ignored) urfall/ · le2i/ · fer2013/ · self_recorded/ · sessions/
├── models/           (git-ignored) exported .onnx / .pt
├── tests/            unit/ · replay/ · scenarios/
├── docs/             context/ · journal.md · proposal/ · srd/ · diagrams/ · final_report/
├── CLAUDE.md · requirements.txt · README.md · .env.example · Makefile
```

## LLM client (§7.1, §7.3)

All provider calls go through `ally/agent/llm_client.py` (`LLMClient`): `dialogue_turn(state, history, tools)`
streamed so TTS starts on the first sentence; `verify_keyframe(jpeg, state) -> VerifyResult` and
`classify_reply(transcript) -> ReplyLabel` using structured outputs; tool schemas declared with `strict: true`;
the stable persona prompt is prompt-cached. Model IDs and prices are fixed in the SRD from the provider's live
pricing page — never hard-coded outside `config.py`. Dialogue route effort is a config value tuned in Sprint 4.
