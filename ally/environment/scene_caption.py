"""Local VLM scene caption (FR-21) → SceneCaption at a very low rate.

Never an input to VERIFY; the text stays on the device and never reaches the cloud LLM (ADR-018).
Enabled only if the S1 VRAM table allows (`ally_scene_caption_enabled`, P-55).

Sprint: S5 (if MVP). See docs/context/architecture.md and methods.md §6.4.
"""
