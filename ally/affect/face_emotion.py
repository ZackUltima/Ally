"""Face detect on the pose head region → quality gate → FER CNN → AffectSample.

One channel of the multi-modal mood estimate (ADR-013); never an alert on its own. The gate checks face
size, near-frontal pose and detector confidence; a face that fails it gives face_visible=False and no
valence — never a guess. Crops stay in RAM and are never written or sent (ADR-015). Gate numbers and the
sample rate come from config (SYNTHESIS, decisions.md "Numbers for ADR-013/014/017").

Sprint: S2 (model), S5 (live worker). See docs/context/architecture.md and methods.md §6.2.
"""
