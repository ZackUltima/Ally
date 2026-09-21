# Schedule & work breakdown

Source: `Ally_Project_Plan.html` v1.1 §2.3 (course phases), §8 (sprints, weekly rhythm), §13 (milestone checklists).
Course dates are from the CAT405 Guidelines v01 2026/27 (VERIFIED · PRIMARY). Sprints are two weeks.

## Hard deadlines
| Date | Deliverable |
|---|---|
| Sun 11 Oct 2026 | project confirmed in SPTA (bidding 28 Sep – 11 Oct) |
| Sun 18 Oct 2026 | initial proposal → eLearn@USM + supervisor |
| Sun 29 Nov 2026 | SRD report; presentation 30 Nov – 13 Dec |
| 8 – 21 Mar 2027 | Milestone 2 progress demo (rehearsal-ready by 7 Mar; Hari Raya 10–11 Mar) |
| Sun 16 May 2027 | final report |
| 17 – 30 May 2027 | final demo; PIXEL 7 – 13 Jun if selected |

## Next four weeks
- **Week 0 (21–27 Sep):** SPTA supervisor shortlist + profile; email 2–3 supervisors a one-page concept;
  Python 3.11 venv + CUDA PyTorch + Ultralytics + OpenCV; YOLO-pose on webcam → FPS/VRAM; repo skeleton
  (`CLAUDE.md`, `docs/`, `.gitignore`, `Makefile`); download UR Fall; `extract_keypoints.py`.
- **Weeks 1–2 (28 Sep – 11 Oct, bidding):** bid day 1; 12–15 literature notes; rule-based fall baseline on
  UR Fall keypoints (→ "Status of the Project"); draft proposal sections.
- **Week 3 (12–18 Oct):** finish proposal (Proposed Solutions, Uniqueness table, Expected Outcomes, SDG,
  APA refs, Gantt); supervisor comments by Fri; upload Sat.
- **Weeks 4–5 (19 Oct – 1 Nov):** API key; first STT → LLM → TTS loop, measure round-trip and first-audio
  latency at two effort settings; GRU fall classifier v0 vs rule baseline; address vetting feedback.

## Sprints (exit criteria are tests where possible)
| Window | Sprint | Tasks | Exit criteria |
|---|---|---|---|
| 2 – 29 Nov | SRD report | interviews with 3–5 carers (notes only); use cases; FR/NFR; architecture + sequence diagrams; FSM spec; data model; wireframes; test plan; ethics | SRD uploaded 29 Nov |
| 30 Nov – 13 Dec | SRD presentation | slides; rehearse twice; 2-min live demo (pose + talking prototype) with backup video | delivered |
| 14 – 27 Dec | **S1 Perception** | capture → pose → keypoints ≥ 15 FPS over bounded queues; ring buffer; presence; inactivity timer; YOLO vs MediaPipe benchmark, freeze choice; **`record_session.py` + `--replay`**; **per-model VRAM table** | live keypoints + inactivity events; FPS ≥ 15; a recorded session replays to identical events; VRAM table filled |
| 28 Dec – 10 Jan | **S2 Fall classifier** | features; GRU + tree baselines; LOSO-CV on UR Fall (+ Le2i); τ_fall; ONNX; integrate | recall ≥ 0.90 mean over folds with CI; results table v1 |
| 11 – 24 Jan | **S3 Event engine + notify** | FSM with a unit test per transition (ADR-001 rules); SQLite schema; Telegram bot with keyframe + ack + chat-ID allowlist; retention job (tested); VLM verification with structured output | staged fall → Telegram within 60 s; FSM tests green; retention test green |
| 25 Jan – 7 Feb (CNY) | **S4 Speech + agent** | VAD/STT/TTS; persona; strict tools as requests; reply classifier; streaming loop; effort chosen by latency measurement; offline fallback + local alarm; first 10 scenarios incl. injection | `VERIFY` runs end-to-end by voice; stand-down works; injection cases pass |
| 8 – 21 Feb | **S5 Emotion + fusion** | FER fine-tune + eval (*fallback: pretrained weights*); mood EMA; `COMPANION_CHAT` with cooldowns; dashboard v1 (*may slip to S6*) | negative mood → proactive chat; FER table |
| 22 Feb – 7 Mar | **S6 Integration + hardening** | 4 h soak; resilience test (network pull, key kill); adversarial review + `/security-review`; 4-scenario demo script; no-internet demo fallback; slides; buffer | demo rehearsed twice without intervention |
| 8 – 21 Mar | **M2 progress demo** | demo MVP against SRD; capture examiner feedback | delivered; feedback in backlog |
| 22 Mar – 4 Apr | S7 Feedback + stretch pick | fix feedback; at most two STRETCH items (suggest FR-10, FR-11) only if MVP green | backlog burned; stretch frozen |
| 5 – 18 Apr | S8 Evaluation runs | final evaluations with fixed seeds; ablations (rule/trees/GRU; ± verifier); 30 scenarios × 5; latency + resource profile; cost log | all result tables final |
| 19 Apr – 2 May | S9 User testing + polish | 3–5 volunteers, consent, scripted sessions, SUS; dashboard polish; README + install guide | usability results; `v1.0-rc` tagged |
| 3 – 9 May | S10 Code freeze + report | freeze; chapters; appendices (manual, consent template, test logs) | draft to supervisor by 7 May |
| 10 – 16 May | M3 report | incorporate comments; upload | submitted 16 May |
| 17 – 30 May | M3 demo | rehearse; final demo; PIXEL poster/video if selected | delivered |

**Slip watch (R14):** S4–S5 at 15–20 h/week. The talking prototype is already due in Weeks 4–5 to de-risk S4;
S5 falls back to pretrained FER; dashboard may move into the S6 buffer. Nothing new after 7 Mar.

## Weekly rhythm
- Monday 30 min: pick sprint tasks; update backlog.
- Fortnightly 30-min supervisor meeting with a one-page status: done / blocked / next / decisions needed.
- Friday: commit, tag, five lines in `docs/journal.md`.
- Time budget 15–20 h/week; guard 2 h/week for literature and report writing.
