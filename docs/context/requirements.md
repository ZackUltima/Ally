# Requirements & scope

Source: `Ally_Project_Plan.html` v1.1 §1 (charter, success criteria) and §4 (user stories, FR, NFR).
Changed by ADR-001 (FR-6), ADR-002 (NFR-1), ADR-003 (NFR-5) — see `decisions.md`.

## Scope

**In (MVP):** single fixed webcam, one person, one living-area room; fall + prolonged-inactivity detection;
facial-expression mood cue; proactive spoken check-in and companionship via cloud LLM; verify-then-escalate
guardian notification on Telegram; local event log + LAN guardian dashboard.
**Out:** medical diagnosis or device claims; multi-camera/multi-person; bathrooms and bedrooms; commercial hardware.

MVP = must work end-to-end for the March 2027 progress demo. STRETCH = only after every MVP exit criterion is green.

## Success criteria (May 2027) — SYNTHESIS unless noted

| Target | Measure |
|---|---|
| ≥ 0.90 | fall recall, mean over leave-one-subject-out folds on UR Fall (+ Le2i), with 95 % CI |
| ≤ 1 / hour | *suspected* events (entering `VERIFY`) on self-recorded daily-activity footage |
| ≤ 1 / day | *guardian-level* false alerts on the same footage |
| ≤ 2 s | fall → `SUSPECTED_FALL` on the RTX 2050 |
| ≤ 3 s | `VERIFY` entry → first spoken audio |
| ≤ 60 s | confirmed event → guardian notified (incl. verification window) |
| ≥ 4 / 5 | mean usability rating, 3–5 volunteers, scripted scenarios |

## User stories

| ID | As a… | I want… | So that… | Tag |
|---|---|---|---|---|
| US-1 | monitored person | Ally to ask if I'm OK when I fall | I get help even if I can't reach a phone | MVP |
| US-2 | monitored person | a check-in after a long period of no movement | a collapse is noticed | MVP |
| US-3 | monitored person | Ally to notice when I look sad and start a gentle conversation | I feel less alone | MVP |
| US-4 | monitored person | to say "I'm fine" and have it stand down | false alarms don't bother my family | MVP |
| US-5 | guardian | a notification with a snapshot and what Ally heard | I can judge urgency instantly | MVP |
| US-6 | guardian | a dashboard with today's events and a mood trend | I can spot slow changes | MVP |
| US-7 | guardian | hazard warnings (stove on, object on floor) | accidents are prevented | STRETCH |
| US-8 | monitored person | Ally to learn my routine and notice deviations | unusual days get a check-in | STRETCH |
| US-9 | monitored person | an on-screen avatar while Ally talks | interaction feels natural | STRETCH |

## Functional requirements

| ID | Requirement | Tag |
|---|---|---|
| FR-1 | Capture webcam frames continuously at ≥ 15 FPS (640×480) and run person + pose detection in real time. | MVP |
| FR-2 | Classify a fall from a sliding window of pose keypoints; emit `SUSPECTED_FALL` with confidence. | MVP |
| FR-3 | Detect prolonged inactivity (no significant keypoint motion for configurable T while a person is present); emit `INACTIVE`. | MVP |
| FR-4 | Estimate facial-expression class per face detection; maintain a smoothed mood estimate. | MVP |
| FR-5 | On any event, verify: send one keyframe + state summary to the VLM (structured JSON answer) and, in parallel, start a spoken check-in with a response timeout. | MVP |
| FR-6 | Dialogue agent converses via speech (STT → LLM → TTS), holds short-term context, and can call `request_escalation` (a request the event engine approves/rejects — the model never notifies the guardian directly), `log_event`, `get_current_state`, `request_stand_down`. In `VERIFY` the model's only decision is to label the reply `fine` / `help` / `unclear` (structured output); the FSM makes the transition. (ADR-001) | MVP |
| FR-7 | Escalate via Telegram with event type, timestamp, keyframe (optionally blurred) and transcript when verification confirms or the person is unresponsive. | MVP |
| FR-8 | Persist events, transcripts and mood samples to local SQLite; expose them on the guardian dashboard. | MVP |
| FR-9 | Proactively start a companionship conversation when mood is negative for > N min, at most once per cooldown. | MVP |
| FR-10 | Scene-hazard description from the VLM on a low-frequency schedule. | STRETCH |
| FR-11 | Learn a per-hour activity baseline over 14 days; flag statistically unusual days. | STRETCH |
| FR-12 | Animated / lip-synced avatar during speech. | STRETCH |
| FR-13 | Voice-tone emotion channel fused with the facial channel. | STRETCH |

## Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-1 | **Latency** (ADR-002): fall → `SUSPECTED_FALL` ≤ 2 s (window + inference); `VERIFY` entry → first spoken audio ≤ 3 s (canned TTS, no LLM turn); LLM dialogue turn → first audio ≤ 3 s at the effort setting chosen in Sprint 4 (current Opus-tier models think by default at effort `high` — measure, lower effort for the dialogue route if needed); confirmed → guardian ≤ 60 s. |
| NFR-2 | **Privacy**: no continuous video storage; frames processed in memory; only event keyframes retained (7 days); only text + event keyframes leave the device (LLM API, Telegram). Consent from every recorded person. |
| NFR-3 | **Resilience**: if internet/API is down, perception and inactivity/fall rules keep running; scripted offline check-in + queued notification retry replace the LLM. Unanswered check-in while offline → local audible alarm. Limitation (no remote person reached until reconnect) stated in ethics. |
| NFR-4 | **Resource budget**: RTX 2050 (4 GB) at ≥ 15 FPS with all MVP models loaded; GPU < 3 GB. Per-model VRAM/CPU table measured in Sprint 1 is the budget of record. Pose + FER on GPU; faster-whisper int8 and Piper on CPU. 10 s ring buffer at 480p ≈ 140 MB. |
| NFR-5 | **False-alarm budget** (ADR-003): suspected events ≤ 1/h of daily-activity footage; guardian-level false alerts ≤ 1/day. The second number decides whether the guardian keeps it on. |
| NFR-6 | **Cost**: cloud API spend ≤ RM 100/month during development; hard spend limit set in the provider console. |
| NFR-7 | **Reproducibility**: one command to run the app, one per experiment; pinned dependencies; fixed seeds; a recorded session replays to identical events. |
