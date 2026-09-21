# Evaluation & testing

Source: `Ally_Project_Plan.html` v1.1 §10. Changed by ADR-002/003 (latency, FA tiers) and ADR-004 (replay level).

## Levels

| Level | What | When | Pass criteria |
|---|---|---|---|
| Unit | FSM transitions (every row of the §5.1 table), feature functions, tool handlers, retention job (pytest) | every sprint; CI on push | all green; every FSM transition covered |
| **Replay** | recorded sessions (falls, look-alikes, ordinary activity) through perception → fusion → agent with the webcam replaced by the recording; expected event sequence asserted. **This is the self-verification loop Claude Code runs after every change.** | every sprint from Sprint 1; CI | every session yields its expected events |
| Model / dataset | fall classifier, leave-one-subject-out on UR Fall (+ Le2i); FER on FER-2013 test; ablations | Sprints 2, 5, 8 | recall ≥ 0.90 mean over folds with CI; fixed seeds, reproducible |
| VLM verification | 100 labelled keyframes (frozen; second rater on 30); agreement, latency, cost | Sprints 3, 8 | agreement ≥ 0.90; inter-rater agreement reported |
| Dialogue scenarios | 30 scripted situations incl. injection cases, each run 5× with the model ID pinned; auto-check the FSM decision; 2 raters score wording | Sprints 4, 8 | escalation-decision accuracy ≥ 0.90; mean rubric ≥ 4/5 |
| Integration / latency | fall → `SUSPECTED_FALL`; `VERIFY` → first audio; event → Telegram; FPS and VRAM under full load; 4 h soak | Sprints 6, 8 | NFR-1 (both numbers), NFR-4 met; no crash |
| Staged live trials | 10 staged falls on a mattress by consenting adults; 10 look-alikes (lying on sofa, picking objects, sitting on floor); 2 h ordinary activity | Sprints 8–9 | ≥ 9/10 falls caught; suspected FA ≤ 1/h; guardian-level FA ≤ 1/day (extrapolated) |
| Usability | 3–5 volunteers play monitored person and guardian; SUS-style questionnaire + interview | Sprint 9 | mean ≥ 4/5; issues logged |
| Resilience | pull the network mid-event; kill the API key; check fallback, local alarm, retry on reconnect | Sprint 6 | offline check-in speaks; alarm sounds if unanswered; notification delivered on reconnect |

## Safety rules for staged falls
Adults only; thick mattress/crash mat; no falls from standing height for anyone with a health condition; a
second person present. No children recorded at any stage (see `ethics.md`).

## Harness contracts
- `tests/replay/<session>/expected_events.json` — ordered list of `(type, t_min, t_max)`; the test asserts the
  replayed engine emits exactly these within tolerance. Sessions are recorded with `scripts/record_session.py`.
- `scripts/run_scenarios.py` — reads `tests/scenarios/*.yaml` (state, scripted utterances, expected FSM decision),
  drives the agent with a fake speech layer, reports decision accuracy and latency distribution; `--repeat 5`.
- Experiments write to `results/<experiment>/<date>/` with seed, model ID, git SHA, and a CSV of metrics.

## Result tables for the final report
1. Fall detection: rule vs trees vs GRU — recall (mean ± CI), specificity, F1, suspected FA/h, latency.
2. Effect of VLM verification: suspected → guardian-level alerts; false alarms removed; true events lost.
3. FER: accuracy, macro-F1, confusion matrix.
4. Dialogue: scenario accuracy over 5 runs, rubric means, latency distribution at the chosen effort.
5. Resource profile: FPS, per-model VRAM, CPU, API cost per day.
6. Usability: questionnaire summary and quotes.
