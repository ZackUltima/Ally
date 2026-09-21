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

## Sprint 0 tooling notes (2026-09-21) SYNTHESIS
- **Python 3.11 via `uv`**: the machine has only 3.13 and no admin-free installer; `uv python install 3.11`
  + `uv venv` gives a reproducible 3.11 venv and `uv pip compile --universal` produces the pinned `requirements*.txt`
  from hand-edited `requirements*.in` (universal, because the dev box is Windows and CI is Linux). Three tiers: `requirements-core.in` (pydantic, numpy — enough for
  unit + replay tests, so CI stays CPU-only and fast), `requirements.in` (full runtime incl. CUDA torch),
  `requirements-dev.in` (ruff, pytest, pre-commit). Exact pins are committed once the first resolve succeeds.
- **`tasks.ps1` mirrors the `Makefile`** target for target because GNU make is not installed on the dev
  laptop; the Makefile remains the contract named in CLAUDE.md.
- **`EventType.SUPPRESSED`** added to `contracts.py` for a `SUSPECTED_*` trigger that fails the 3 s debounce
  ("log as suppressed" in the §5.1 table) so the suppression is a first-class, countable event for the
  ADR-003 false-alarm accounting. No new threshold.
- **`config.py` carries only documented numbers** (640×480, 15 FPS, 10 s ring buffer, 1.5 s window, 30 min
  inactivity, 3 s debounce, 20 s listen × 2 prompts, 10 min confirmed timeout, 5 min cooldown, 5 s mood
  sample, 7-day retention, 127.0.0.1 bind). `ally_fall_threshold` defaults to `None` until Sprint 2's LOSO
  run chooses τ_fall. Dashboard port 8000 is a plain default, not a target.

## Week 0 perception notes (2026-09-22) SYNTHESIS
- **Feature conventions (`ally/perception/fall_features.py`).** The §6.1 step 2 feature set is implemented
  as six per-frame columns in a fixed order (`FEATURE_NAMES`): torso angle in degrees from vertical
  (0° upright, 90° horizontal, > 90° head below hips), bbox width/height, hip height normalised as
  `1 − hip_y / H` (0 = bottom of frame), hip vertical velocity as `d(hip_y / H)/dt` in frame-heights per
  second with **positive = downward** (so the plan's "vertical velocity > v_thr" reads naturally), its
  derivative, and mean keypoint confidence. Derivatives are central finite differences on the real
  timestamps (one-sided at window edges). Missing joints (conf ≤ `min_conf`, default 0 = only undetected
  joints) and absent frames become NaN and propagate to neighbouring derivatives; window statistics are
  NaN-aware. Joint indices exist only in one COCO-17 map; MediaPipe gets its own map if chosen in Sprint 1.
- **Rule baseline parameters (`RuleFallDetector`).** The plan's rule fixes 60° and 2 s but not `v_thr` nor
  what "low hip height" means, so both are **required constructor arguments with no default**, like
  `ally_fall_threshold`; Sprint 2 chooses them on UR Fall and records them here. The rule needs one more
  bound the plan does not state — how long after the impact the hip may take to reach the floor — and reuses
  the 1.5 s feature window (`settle_s`) for it rather than inventing a new number. Detector output is a
  per-frame score in [0, 1] (1.0 once per fall) behind the same `FallDetector` protocol the GRU will use.
- **UR Fall layout and the subject map.** `scripts/prepare_urfall.py` fetches only cam0 RGB (≈ 58 MB per
  sequence, ≈ 4 GB total) over HTTPS from `fenix.ur.edu.pl` (host allow-listed), keeps the authors' CSVs
  verbatim and merges their first three columns into `labels.csv` (label 1 = lying, 0 = falling,
  −1 = not lying, per the dataset page). The dataset does **not** publish which subject performed
  each sequence, so `subjects.csv` is written as a blank template to be filled by hand (by viewing the
  sequences) — never guessed. Until it is filled, leave-one-subject-out CV cannot run; the fallback is
  grouped CV over sequences, and the SRD must say which one was used.
- **Image-sequence frame rate — ASSUMED.** The dataset page (fetched 22 Sep) gives sampling rates only for
  the accelerometers (60 Hz / 256 Hz), not for the RGB frames. `extract_keypoints.py` therefore timestamps
  UR Fall PNG folders at an **assumed 30 FPS** (Kinect RGB nominal; `--fps` overrides). Because `ts` scales
  `hip_vy`/`hip_ay`, any `v_thr` chosen in Sprint 2 is only valid for that rate; the SRD must state the
  assumption or replace it with a verified value (e.g. from the authors' paper or a sequence's known duration).
- **Label join verified.** The page defines label −1 = not lying, 1 = lying on the ground, 0 = "temporary
  pose, when person is falling", and says the authors exclude 0-frames from classification — Sprint 2 should
  do the same and say so. `frame_id` in the npz is the file-name number; on `adl-01` the PNGs run 1–150 while
  the CSV covers 6–150 (144 rows) and every CSV frame has a PNG, so the join is by id and exact. Position-based
  joins would be off by up to six frames on ADL sequences.
