"""Multi-modal mood estimate (FR-18, ADR-013) → PersonState.mood / mood_label.

Late fusion over a window of facial valence (AffectSample), speech-text sentiment (SpeechSentiment) and
movement level vs the person's baseline. Pure: no I/O, time passed in. A missing channel drops out; no
channel → MoodLabel.UNKNOWN. A low estimate may start COMPANION_CHAT (event_engine) but never creates a
Notification or an escalation. Window, weights and thresholds come from config (SYNTHESIS, decisions.md
"Numbers for ADR-013/014/017"); the function takes them as parameters so tests pass their own.

Sprint: S3. See docs/context/architecture.md and methods.md §6.2.
"""
