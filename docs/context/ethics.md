# Ethics, privacy, safety & security (CLO 4)

Source: `Ally_Project_Plan.html` v1.1 §12, §5.2 and risk R13. Applies to every session that touches
recording, storage, the cloud boundary, notifications or user-facing text.

## Consent and participants
- Every person who appears on camera during development or testing signs a consent form (purpose, what is
  stored, for how long, right to withdraw). Forms kept with project records. Template in `docs/final_report/`.
- **No children** are recorded or used as test subjects at any stage; the child use case is future work only.
- Ask the supervisor whether school ethics clearance is needed before Sprint 9 user testing (R6).

## Data minimisation
- No continuous video recording; frames processed in RAM; only one event keyframe persisted, 7-day retention,
  deletable from the dashboard; face-blur option (on by default) for guardian messages.
- `scripts/record_session.py` stores keypoints, audio and events — never raw frames.
- Living areas only; bathrooms and bedrooms are out of scope by design.

## Cloud boundary
- Only text (state JSON, conversation) and one keyframe per event go to the API provider; document the
  provider's data-retention terms in the report; `ALLY_TEXT_ONLY_VERIFY=true` disables image upload.

## Safety claims and limitations — stated plainly
- Ally is a research prototype, **not a medical or emergency device**; the report, README and the first-run
  spoken explanation say so (cf. ElliQ's "not a medical device" statement [3]).
- **Offline:** Ally still detects, checks in and sounds a local alarm, but **no remote person is notified until
  connectivity returns**. Ally supplements human contact and emergency services; it does not replace them.
- **Emotion cue:** expression ≠ emotion; FER datasets are not representative of all populations; the cue only
  triggers a gentle conversation and never an alert on its own (R11).
- Report performance across lighting conditions.
- Dashboard audience: LAN-only, so co-resident carers or a guardian visiting the home; the remote guardian
  receives the daily summary and mood trend through Telegram.

## Transparency to the monitored person
- Visible indicator when Ally is listening/speaking; spoken explanation on first run; a physical "pause
  monitoring" control that the FSM honours.

## Security checklist (verify, don't assume)
- [ ] API keys only in `.env` (git-ignored); `.env.example` documents names; hard spend limit set in the console.
- [ ] Telegram handler rejects every chat ID except `TELEGRAM_GUARDIAN_CHAT_ID` (test with a second account).
- [ ] Dashboard binds to the LAN interface with token auth or HTTPS — not basic auth over plain HTTP.
- [ ] Retention job actually deletes keyframes and rows after 7 days (test with a fake clock).
- [ ] Image-upload kill switch honoured on every VLM call path.
- [ ] STT output is untrusted input: the model can only label a reply; it cannot suppress an escalation (ADR-001, R13).
- [ ] Before Sprint 6: adversarial review in a fresh session ("find what's wrong") + `/security-review`.
