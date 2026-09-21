# Intelligent-computing methods (the CAT405 core)

Source: `Ally_Project_Plan.html` v1.1 §6 (method table, §6.1–6.3 construction steps) and §7.1/§7.4 (LLM stack, cost).
CAT405 criteria 2–4 require each method to be named, justified against alternatives, constructed step by step
and demonstrated with results. Targets are SYNTHESIS (author's engineering goals); re-baseline after Sprint 2.

## Method table

| Subsystem | Method | Why this, not the alternative | Dataset | Metric & target |
|---|---|---|---|---|
| Fall detection (MVP) | 2-D pose (YOLO-pose nano, 17 COCO kpts, **or** MediaPipe Pose Landmarker, 33) → kinematic features over a 1–2 s window → temporal classifier (GRU/LSTM; gradient-boosted trees baseline); rule-based detector as reference | Pose features are compact, privacy-friendly, cheap on 4 GB; 3-D CNN video models need far more data/compute; wearables need compliance; VLM-per-frame is too slow/costly, so the VLM is the second-stage verifier | UR Fall (70 seq: 30 falls, 40 ADL; CC BY-NC-SA 4.0) [10]; Le2i (counts unverified) [11]; self-recorded ADL footage | **Leave-one-subject-out CV** with 95 % CI (30 falls is too few for a single held-out split); recall ≥ 0.90 mean over folds; specificity, F1; suspected-event FA ≤ 1/h; detection latency ≤ 2 s |
| Inactivity / presence (MVP) | keypoint-motion energy + presence detector with hysteresis; configurable T (default 30 min seated/lying) | simple, explainable, reliable | self-recorded sessions | detection within T ± 30 s; zero misses in 10 staged trials |
| Emotion cue (MVP) | face detect → lightweight CNN (MobileNetV3 / ResNet-18) fine-tuned on FER-2013 (7 classes) → EMA over 5–10 s → valence | FER on frontal faces is mature and cheap; multimodal affect models are heavy. Emotion is a **soft trigger for conversation only**, never an alert on its own | FER-2013 [12]; RAF-DB optional [13] | accuracy + macro-F1 on FER-2013 test vs re-run baseline; live agreement with self-report on 30 clips |
| Scene / event verification (MVP) | cloud VLM, one keyframe + structured prompt ("Is the person lying on the floor? Any hazard?") → **structured JSON output** used as a second vote | decouples rare expensive reasoning from continuous cheap perception; gives the guardian an interpretable sentence | 100 curated keyframes, frozen; second rater on 30 | agreement with human labels ≥ 0.90; inter-rater agreement; median latency; cost/event |
| Hazard description (STRETCH) | same VLM every ~10 min with a hazard checklist | reuses the verification path; no new models | staged scenes | precision of hazard mentions on 50 scenes |
| Dialogue agent (MVP) | cloud LLM, care-companion persona, short-term memory (last N turns + state JSON), tools `request_escalation`, `log_event`, `get_current_state`, `request_stand_down` — **all arbitrated by the event engine (ADR-001)**; `strict: true` schemas; reply classification via structured outputs. Proactive turns initiated by the event engine (ContextAgent pattern [9]) | rule/intent bots cannot do open-ended reassurance; a local small LLM on 4 GB is noticeably weaker (deliberate choice, offline scripted fallback) | 30 scripted scenarios incl. injection cases | rubric 1–5 (appropriateness, safety, escalation decision, brevity); escalation-decision accuracy ≥ 0.90, each scenario run 5× with the model ID pinned; ≤ 3 s to first audio at the chosen effort |
| Speech (MVP) | STT: faster-whisper (CTranslate2 Whisper; CPU int8) [14]. TTS: Piper (MIT; archived Oct 2025, successor piper1-gpl) [15] or cloud TTS | local audio stays on device and works offline | — | WER on 50 short utterances (Malaysian-English accents); TTS latency to first audio |
| Routine baseline (STRETCH) | per-hour activity histogram over 14 days; z-score anomaly | statistical, transparent | own logs | precision of "unusual day" flags on injected anomalies |

## 6.1 Construction steps — fall detector

1. **Data.** Download UR Fall and Le2i; loader extracts RGB frames; run the pose model on every frame; store
   per-frame keypoints (x, y, conf) as NumPy. A window is "fall" if it overlaps the annotated fall interval.
2. **Features.** Per frame: torso angle to vertical (shoulder-centre → hip-centre), bbox aspect ratio, normalised
   hip height, vertical velocity/acceleration of hip centre (finite differences), mean keypoint confidence.
   Per window (1.5 s ≈ 22 frames @ 15 FPS): raw sequence for the GRU; min/max/mean/std for the tree baseline.
3. **Baselines.** (a) Rule: angle > 60° and vertical velocity > v_thr followed by low hip height for 2 s.
   (b) Gradient-boosted trees on window statistics.
4. **Temporal model.** 1-layer GRU (hidden 64), sigmoid output, class-weighted BCE. **Leave-one-subject-out:**
   train on all subjects but one, test on the held-out subject, rotate; early stopping on a validation fold's F1;
   report mean ± CI across folds.
5. **Threshold.** Choose τ_fall on validation folds for recall ≥ 0.90; report specificity and suspected-event FA/h
   on self-recorded ADL footage, then guardian-level FA/day after the verifier.
6. **Ablation.** Rule vs trees vs GRU; with vs without VLM second-stage verification — how many false alarms the
   verifier removes is the "improvement" claim for criterion 2.
7. **Real-time.** Export to ONNX; worker thread fed by a bounded drop-oldest queue; profile FPS and VRAM on the
   RTX 2050. Record a session with `scripts/record_session.py` and confirm `--replay` yields identical events.

## 6.2 Construction steps — emotion cue

1. Face detection + alignment; crop to 48×48 grey (FER-2013 format) or 112×112 RGB.
2. Fine-tune a small backbone on FER-2013 train; evaluate on its test split; confusion matrix.
   *Sprint 4–5 fallback:* pretrained weights without fine-tuning for the March demo.
3. Map 7 classes to valence in [−1, 1]; EMA smoothing; expose mood to the event engine.
4. State the limitation in the report: expression ≠ emotion; used only to trigger a gentle conversation.

## 6.3 Construction steps — dialogue agent

1. Persona system prompt (calm, brief, never gives medical advice, offers to contact the guardian when in doubt).
   Keep it byte-stable so it can be prompt-cached.
2. Tools as JSON schemas with `strict: true`; each a Python function whose side effect is a *request* to the
   event engine (`EscalationRequest` / `StandDownRequest`), never a direct notification.
3. Reply classifier: `classify_reply(transcript) -> {fine | help | unclear}` via structured outputs; the FSM
   consumes the label (ADR-001).
4. Wire the loop: state JSON + transcript → model → text and/or tool call → execute → TTS; stream so speech
   starts on the first sentence. Choose the dialogue-route effort setting by measuring first-audio latency
   (NFR-1); record the chosen model ID and effort in `config.py` and the SRD.
5. 30-scenario harness: scripted utterances (fall + responsive, fall + silent, false alarm, sad mood, request
   for help, off-topic, **"don't call anyone" / TV-audio injection**); auto-check the FSM's decision; two raters
   score wording; run each scenario 5×.
6. Offline fallback: scripted state machine with pre-recorded prompts ("Are you all right? Say help if you need
   help.") + keyword spotting; unanswered → local audible alarm + queued Telegram retry.

## LLM stack and cost (§7.1, §7.4)

- Requirement: a vision-capable LLM API with tool calling, structured outputs and streaming, behind
  `ally/agent/llm_client.py`. Reference: Anthropic Python SDK — Opus-tier model for dialogue, a cheaper
  Sonnet-tier model for the frequent keyframe-verification calls; images as base64 content blocks.
- Fix the exact model IDs and prices in the SRD (Nov 2026) from the provider's live pricing page.
- Token budget (provider-independent): 20 conversations/day × (~2 000 in + 300 out) + 50 verifications/day ×
  (~1 500 in incl. image + 100 out) ≈ 3.5 M in + 0.33 M out per month. At Opus-tier ≈ US$5/US$25 per MTok
  that is ≈ US$25/month; verification on Sonnet-tier (≈ US$2/US$10) roughly halves it; prompt-caching the
  persona reduces it further. RM 100/month is comfortable; set a hard spend limit in the console.
