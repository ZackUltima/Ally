"""AudioSet-pretrained sound tagger on CPU over the microphone stream → allow-listed SoundEvent (FR-20).

Hazard labels start a spoken check-in via SUSPECTED_HAZARD (ADR-017); logged labels only go to the daily
summary; "Thump, thud" is fall evidence only. This module only emits events. Audio windows stay in RAM.

Sprint: S4. See docs/context/architecture.md and methods.md §6.4.
"""
