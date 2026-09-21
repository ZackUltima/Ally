# Decisions (ADR log)

One paragraph per decision: context → decision → consequence. Anything tagged SYNTHESIS is the author's
engineering judgement, not a sourced fact. Add a new entry before changing any threshold, target or contract.
Source: `Ally_Project_Plan.html` v1.1, §0 "Changes in v1.1".

## ADR-001 — The event engine owns every escalation decision (2026-09-21)
**Context.** Plan v1.0 §5.1 exited `VERIFY` on "person says they are fine (LLM interprets)" and FR-6 gave the
model `notify_guardian` and `stand_down` as tools. A non-deterministic model fed speech-to-text of uncontrolled
room audio would therefore hold the safety decision; misheard speech, a TV, or a spoken "don't call anyone"
could suppress a real alert.
**Decision.** `ally/fusion/event_engine.py` computes every transition. In `VERIFY` the model's only output is a
structured label for the reply — `fine` / `help` / `unclear`. `fine` **and** no VLM fall confirmation →
`STOOD_DOWN`; `help`, `unclear`, silence after two prompts, or VLM + posture agreeing on a fall → `CONFIRMED`.
The tools become `request_escalation` / `request_stand_down`; the engine approves or rejects them. Only
the engine can create a `Notification`.
**Consequence.** Escalation logic is deterministic and unit-testable without an API key. The scenario harness
gains injection cases (R13). Slightly more code in the FSM; the persona prompt loses the ability to "decide".

## ADR-002 — Latency target split into two numbers (2026-09-21) SYNTHESIS
**Context.** v1.0 promised fall → first spoken check-in ≤ 2 s, but the classifier uses a 1.5 s window and the
`SUSPECTED_*` state adds a 3 s debounce before `VERIFY`, where speech starts. The floor was ≈ 5 s.
**Decision.** NFR-1 is now: fall → `SUSPECTED_FALL` ≤ 2 s (window + inference); `VERIFY` entry → first audio
≤ 3 s using a canned TTS prompt (no LLM turn); LLM dialogue turn → first audio ≤ 3 s at the effort setting
chosen in Sprint 4; confirmed → guardian ≤ 60 s. Current Opus-tier models run adaptive thinking at effort
`high` by default, so measure and lower effort for the dialogue route if needed. Streaming with TTS fired on
the first sentence is required.
**Consequence.** Honest, measurable numbers for the SRD. End-to-end fall → speech is ≈ 5–6 s and is reported
as such.

## ADR-003 — False-alarm budget is two-tiered (2026-09-21) SYNTHESIS
**Context.** v1.0 NFR-5 allowed ≤ 1 guardian notification per hour from non-events — up to 24 a day — which is
exactly the failure R2 warns about (guardians switch it off).
**Decision.** *Suspected* events entering `VERIFY` ≤ 1/h of daily-activity footage (cost: one "are you all
right?"). *Guardian-level* false alerts ≤ 1/day. The verifier and check-in are what convert the first
number into the second; the ablation in §6.1 measures exactly that reduction.
**Consequence.** τ_fall is tuned for recall; precision is bought in `VERIFY`, not in the detector.

## ADR-004 — Record/replay harness is an MVP deliverable (2026-09-21)
**Context.** Nothing in v1.0 let the pipeline run without a live webcam and a person in front of it, so every
change would need a human in the room to verify, and ablations would not be reproducible.
**Decision.** `scripts/record_session.py` writes keypoint streams, audio and the event log (never raw frames)
to `data/sessions/<name>/`; `ally/main.py --replay <name>` swaps the camera for the recording; `tests/replay/`
asserts the expected event sequence per session. Sprint 1 exit criterion; runs in CI on every push.
**Consequence.** Claude Code can self-verify perception/fusion changes; staged falls are recorded once and
reused; the dialogue harness can be driven from replayed events.

## Smaller decisions recorded in v1.1
- **Leave-one-subject-out cross-validation on UR Fall** (30 fall sequences; a single held-out split has too few
  falls for a stable recall). Report mean ± 95 % CI across folds. §6, §6.1.
- **Local audible alarm** when a check-in is unanswered offline; the offline limitation is stated in §12.
- **Per-model VRAM/CPU table** measured in Sprint 1 is the NFR-4 budget of record; STT and TTS on CPU.
- **Capture at 640×480**; 10 s ring buffer ≈ 140 MB (vs ≈ 930 MB at 1080p).
- **Structured outputs** for the VLM verification answer and the reply classifier; `strict: true` on tools.
- **Dialogue scenarios run 5× each** with the model ID pinned; VLM keyframe set frozen with a second rater on 30.
- **Dashboard is LAN-only by design**; the remote guardian gets the daily summary through Telegram.
- **Sprint 4–5 fallback**: pretrained FER weights without fine-tuning for the March demo; dashboard v1 may
  slip into the Sprint 6 buffer.
