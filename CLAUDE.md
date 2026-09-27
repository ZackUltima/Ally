# Ally — CLAUDE.md

Ally is a CAT405 final-year project (USM, 2026/27): a **multi-modal** laptop + webcam + microphone companion for an
older adult living alone. It detects falls and prolonged inactivity, estimates mood from facial expression, speech and
movement, notices household objects and sounds, **talks to the person first**, and escalates to a guardian on Telegram
only after verification. Raw video never leaves the device. Scope: ADR-012 (supervisor feedback, 25 Sep 2026).

Full context lives in `docs/context/` — read only the file(s) relevant to the task (see `docs/README.md`).
The current plan is `Ally_Project_Plan_v1.3.html` (v1.3; v1.2 is kept as the previous version); the context docs are
derived from it. Reading list: `research/` (one .md per topic, title + link).

## Stack
- Python 3.11 in a venv (3.13 is installed on the machine — do not use it; CV/ML wheels lag).
- Vision: OpenCV, Ultralytics YOLO-pose nano **or** MediaPipe Pose Landmarker (decided in Sprint 1), ONNX Runtime (CUDA EP).
  Face detector + a small pretrained FER model (ADR-013); YOLO object detector (ADR-014); a small **local** VLM for
  scene captions only if the Sprint 1 VRAM table allows (ADR-014). All beyond pose run at low sample rates.
- ML: PyTorch (CUDA), scikit-learn, LightGBM. GPU is an RTX 2050 with 4 GB VRAM — small models only.
- Speech/audio: faster-whisper on **CPU int8** (GPU is reserved for vision), Piper TTS, sounddevice, Silero/WebRTC VAD;
  an AudioSet-pretrained sound tagger on CPU (ADR-014).
- LLM (text only, no image to any cloud model — ADR-007/015): Anthropic Python SDK behind `ally/agent/llm_client.py` (`LLMClient` interface — nothing else imports `anthropic`).
- Backend: FastAPI + Uvicorn, SQLModel on SQLite, Jinja2 dashboard. Notifications: python-telegram-bot.
- Tooling: pytest, ruff, pre-commit, Makefile. Pinned `requirements.txt`.

## Layout (see `docs/context/architecture.md` for the full tree)
```
ally/perception  camera.py pose.py fall_features.py fall_model.py presence.py
ally/affect      face_emotion.py (face → quality gate → FER → AffectSample; no identification — ADR-013/015)
ally/environment objects.py sounds.py scene_caption.py (local VLM, VRAM-gated — ADR-014)
ally/fusion      state.py event_engine.py (pure FSM, no I/O) cooldowns.py mood.py (late fusion, pure)
ally/agent       dialogue.py tools.py llm_client.py prompts/persona.md offline_fallback.py
ally/speech      vad.py stt.py tts.py
ally/notify      telegram_bot.py        ally/api  server.py dashboard/ (guardian) + person screen (ADR-009)
ally/storage     db.py models.py retention.py
ally/contracts.py  pydantic models exchanged between modules
ally/main.py       starts workers over bounded queues; --replay <session> replaces the webcam
scripts/  record_session.py extract_keypoints.py train_fall.py eval_fall.py train_fer.py eval_fer.py run_scenarios.py benchmark_fps.py
tests/    unit/ (FSM, features, tools, mood)  replay/ (recorded sessions → expected events)  scenarios/ (30 dialogue cases)
docs/     context/  diagrams/  journal.md  proposal/  srd/  final_report/      research/  reading list (title + link)
```

## Module contracts (typed, in `ally/contracts.py`)
- perception → fusion: `KeypointFrame` (timestamp, keypoints[N,3], bbox, person_present).
- affect → fusion: `AffectSample` (face_visible, valence, class_probs, quality — numbers, never pixels). environment → fusion:
  `SceneObjects`, `SoundEvent`, `SceneCaption`. agent → fusion: `SpeechSentiment`. (ADR-016, proposed — P-51.)
- fusion → agent: `PersonState` (posture, motion, presence, zone, activity, `mood` + `mood_label` fused by `fusion/mood.py`,
  ADR-013) and `Event` (type, confidence, keyframe_id, state).
- agent → fusion: `EscalationRequest` / `StandDownRequest` (reason, urgency). **Requests, not actions.**
- fusion → notify: `Notification` (event, keyframe, transcript). Only `event_engine` may create one.
Each module is testable at its boundary with fixtures; never reach across a boundary to a concrete class.

## Hard rules
1. **The FSM owns escalation.** The LLM classifies a check-in reply as `fine` / `help` / `unclear` and may *request*
   escalation or stand-down; `ally/fusion/event_engine.py` decides. `help`, `unclear` or silence after 2 prompts →
   `CONFIRMED`. A suspected fall never stands down: `fine` + proven up → `FALL_RECOVERED`, a calm message that also
   notifies the guardian (ADR-006). Only an inactivity check-in can stand down, and only on `fine` + proven up (truth
   table in ADR-005/007). "Up" must be proven — standing, or on a seat/bed in the zone map; doubt = down/unknown →
   `CONFIRMED`; standing outside a zone still sends one calm message (ADR-011). No VLM votes in `VERIFY` (ADR-007; the
   local scene caption never does — ADR-014). A mood estimate, an object or a caption never escalates on its own: low
   mood only starts `COMPANION_CHAT` (ADR-013). A hazard sound (smoke/fire alarm, shatter) starts a check-in via
   `SUSPECTED_HAZARD` with the same `VERIFY` cells, and a hazard check-in never ends silently (ADR-017). The model has no tool that sends a Telegram message. (ADR-001)
2. **Never write raw frames to disk.** Frames live in the RAM ring buffer; only one JPEG keyframe per event is
   persisted (7-day TTL). `record_session.py` records keypoints, audio and events — not video.
3. **Privacy boundary (ADR-015).** Local models on the laptop may process frames and face crops **in RAM**; no frame,
   face crop or keyframe is sent to any cloud model, no face crop is written to disk, and nothing identifies the person.
   Only conversation text and a JSON state summary (incl. mood / object / sound labels) go to the LLM; local VLM
   captions never do (ADR-018). One keyframe per event goes to the guardian's Telegram; respect
   `ALLY_GUARDIAN_KEYFRAME=false` (no photo).
4. **Secrets.** Keys in `.env` only; `.env.example` documents them. Never print or log a key.
5. **Numbers are owned by the human.** Do not change a threshold, target (recall, FA/h, latency) or cooldown without
   adding a `SYNTHESIS` note to `docs/context/decisions.md`. Do not invent citations or statistics.
6. **No medical or emergency-device claims** in prompts, UI text or docs. Ally is a research prototype. The mood label
   is an estimate of *expressed* mood ("seemed low today"), never "depression", "distress" or any clinical term; the
   person's button says "I need help", never "emergency". Ally does not detect a stove left on, gas or a wet floor.
7. **Telegram bot** rejects any chat ID other than `TELEGRAM_GUARDIAN_CHAT_ID`. Dashboard binds to LAN only. The person's screen is
   laptop-only (`127.0.0.1`) in the MVP; the phone version (FR-15, STRETCH) needs pairing/auth first (ADR-009, P-46).
8. Capture resolution is 640×480. Queues between workers are bounded, drop-oldest. Perception must stay ≥ 15 FPS.
9. Retention job must be tested to actually delete. Keyframes are deletable from the dashboard.

## Commands (Makefile — create in Sprint 0; targets below are the contract)
- `make test`      pytest `tests/unit` + `tests/replay`
- `make lint`      ruff check + ruff format --check
- `make replay SESSION=<name>`   run a recorded session through perception → fusion → agent, print events
- `make scenarios` run the 30 scripted dialogue scenarios through the agent harness (needs API key)
- `make run`       live app;  `make bench`  FPS/VRAM profile
- Experiments: `make train-fall`, `make eval-fall`, `make train-fer`, `make eval-fer` (fixed seeds, results to `results/`).

## Definition of done for any task
- `make test` and `make lint` green.
- Touched `perception/`, `affect/`, `environment/` or `fusion/` → `make replay` on every session in `tests/replay/` still yields expected events.
- Touched `agent/` → `make scenarios` escalation-decision accuracy unchanged or better; injection cases still pass.
- New decision or number → paragraph in `docs/context/decisions.md`. Five lines in `docs/journal.md`.
- Report what was verified and how; if something was not run, say so.

## Team process (Scrum, adapted to one human + agents) — every agent follows this
Roles: the **human is Product Owner and sponsor** (orders `docs/backlog.md`, owns every number, accepts work).
The **main Claude Code session is the orchestrator** — it is the only thing that can call agents, so it plays the
Scrum-Master-at-the-keyboard: runs the rituals below and calls `agile-coach` before every delegation.
`agile-coach` facilitates and enforces (read-only); `project-manager` tracks the calendar; the other agents are the
developers and specialists. No agent is above another; the backlog is the only source of work.

- **Backlog.** `docs/backlog.md` — Product Goal, current Sprint Goal, ordered items with acceptance criteria and
  status (`todo` → `ready` → `doing` → `review` → `done`). Work not on the backlog is not started; an agent that
  discovers work adds a proposed item (status `todo`, "proposed by <agent>") instead of doing it.
- **Definition of Ready** (an item may enter a sprint only when all hold): (1) one small vertical slice, doable in one
  session; (2) acceptance criteria that are observable — a test, a table, a file, a number with its unit; (3) any
  prerequisite done or explicitly marked as the blocker — ADR for contracts/FSM/privacy, design in `docs/design/`
  for UI, `decisions.md` entry for a number; (4) the agent/owner named; (5) it serves a US/FR/NFR or a course
  deliverable. Not ready → it stays `todo` and the blocker is named.
- **One item `doing` at a time.** Finish or park (back to `ready`, with a note) before starting another.
- **Rituals** (commands in `.claude/commands/`): `/sprint-plan` on the sprint's first Monday — Sprint Goal, pull
  `ready` items, human confirms; `/standup` at the start of every working session — Done since last / Doing now /
  Blocked, then the coach's process check; `/sprint-review` on the sprint's last Friday — each exit criterion
  demonstrated or not, with evidence; `/retro` right after the review — what to keep / change, one process
  action for next sprint, recorded in `decisions.md` (Process notes).
- **Stand-up report format** — every agent ends its work with exactly these five lines so the human and the coach can
  read it in ten seconds: **Item:** B-n · **Done:** what exists now · **Evidence:** tests run / files / numbers
  (say "not run" where true) · **Blocked:** what and by whom · **Next:** the one next step. Followed by the DoD
  checklist, each line done / not done / N/A.
- **Sustainable pace.** 15–20 h/week; 2 h/week protected for literature and report writing; no new scope after
  7 Mar 2027. Ceremonies are short: a stand-up longer than the work it reports is waste.

## Dev team (subagents in `.claude/agents/`) — use sparingly
Default is to do the work in the main session. Delegate only when the row below matches; never run more than two
agents at once; agents do not spawn agents. Read-only roles cost little; builders cost real usage — give them a
scoped task with a test to make green, not "build the feature".

**Always call `agile-coach` first.** Before delegating to any agent in the table below, invoke `agile-coach`
(read-only, cheap). It checks slice size, order (ADR before contracts/FSM, design before UI, builder → QA →
tech-lead, `decisions.md` before a changed number) and CLAUDE.md's definition of done, then reports to the
human what is not done / done incorrectly / done out of order, and emits an *agile brief*. Paste that brief
verbatim into the delegated agent's prompt. If the coach says a prerequisite is missing, do the prerequisite
first — do not skip to the builder.

| When | Delegate to |
|---|---|
| before any other agent — process check + agile brief for it | `agile-coach` (read-only, always first) |
| a change to `contracts.py`, FSM states, queues, LLM client, or the privacy boundary is being considered | `software-architect` (2 options → human decides → ADR) |
| "what should we build next / is X in scope" | `product-owner` (read-only) |
| vague request or examiner feedback needs turning into FR/NFR, SRD text, interview script | `business-analyst` |
| a new screen or message format before any code | `ui-ux-designer` → then `frontend-dev` |
| scoped work in one or two packages with a test | `backend-dev` (core) · `frontend-dev` (dashboard/Telegram UI) |
| a feature spanning perception → storage → api/notify as one slice | `fullstack-dev` |
| training/eval scripts, datasets, LOSO-CV, FER / object / sound evaluation, ONNX export, VRAM table, result tables | `ml-engineer` |
| a builder says "done" on fusion/agent/notify/storage; before milestone demos | `qa-engineer`, then `tech-lead` review |
| Makefile, pinning, pre-commit, CI, tags, demo runbook | `devops-engineer` |
| sprint start/end, supervisor meeting, "are we on track" | `project-manager` |

Do **not** delegate: one-file fixes, questions answerable from `docs/context/`, anything the human is mid-way
through deciding.

## Working with Claude Code on this repo
- One session per task; small vertical slices ("FSM `VERIFY` transitions + tests", not "build the event system").
- Use plan mode before touching the FSM, thresholds, the privacy boundary, or the escalation path; propose 2 options.
- Point the session at the one `docs/context/*.md` it needs; do not paste the whole plan.
- Before Sprint 6 demo hardening, run an adversarial pass in a fresh session: "find what's wrong, especially
  the escalation path, retention and the Telegram handler" — then `/security-review`.
- When a session runs long, ask it for a handoff prompt for the next session.
