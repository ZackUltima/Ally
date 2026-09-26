"""Runtime settings. Keys come from `.env` (documented in `.env.example`) and are never logged.

Every number here is one already fixed in docs/context/ (requirements.md, architecture.md, decisions.md).
Changing a default is a decision: add a SYNTHESIS note to docs/context/decisions.md first (hard rule 5).
Values still to be measured (τ_fall, model IDs, dialogue effort) default to None / placeholders on purpose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- secrets (hard rule 4) ---------------------------------------------------------------
    anthropic_api_key: SecretStr | None = None
    telegram_bot_token: SecretStr | None = None
    telegram_guardian_chat_id: str | None = Field(
        default=None, description="the only chat ID the bot answers"
    )

    # ---- model routing (IDs fixed in the SRD; never hard-coded elsewhere) --------------------
    ally_dialogue_model: str | None = None
    ally_dialogue_effort: Literal["low", "medium", "high"] = "high"  # re-measured in Sprint 4 (NFR-1)

    # ---- privacy switches (NFR-2, hard rule 3) ------------------------------------------------
    # No image ever goes to a cloud model (ADR-007/015). False → the guardian's Telegram message has no
    # photo either.
    ally_guardian_keyframe: bool = True
    ally_blur_face_for_guardian: bool = True
    ally_keyframe_retention_days: int = Field(default=7, ge=1)

    # ---- perception (FR-1, NFR-4, hard rule 8) ------------------------------------------------
    ally_capture_width: int = 640
    ally_capture_height: int = 480
    ally_target_fps: int = 15
    ally_ring_buffer_s: float = 10.0
    ally_fall_window_s: float = 1.5  # methods.md §6.1 step 2
    ally_fall_threshold: float | None = Field(
        default=None, description="τ_fall — chosen in Sprint 2 by LOSO-CV"
    )

    # ---- event engine (architecture.md §5.1) --------------------------------------------------
    ally_inactivity_minutes: float = 30.0
    ally_debounce_s: float = 3.0
    ally_verify_listen_s: float = 20.0
    ally_verify_max_prompts: int = 2
    ally_confirmed_timeout_min: float = 10.0
    ally_stand_down_cooldown_min: float = 5.0
    # ADR-005/006 settings; SYNTHESIS values, accepted provisionally by the human on 23 Sep 2026 (P-39).
    ally_decision_posture_window_s: float = 2.0  # majority vote of posture at the VERIFY decision
    ally_recovery_min_down_s: float | None = Field(
        default=None, description="None = any impact that reaches the floor counts as a fall"
    )

    # ---- mood estimate (ADR-013) — SYNTHESIS values, decisions.md "Numbers for ADR-013/014/017" --
    ally_affect_sample_s: float = 1.0  # seconds between face/FER samples (judgement; re-baseline S1)
    ally_face_min_px: int = 48  # quality gate: minimum face box side (FER-2013 is 48×48, SECONDARY)
    ally_face_max_yaw_deg: float = 30.0  # quality gate: near-frontal (judgement)
    ally_face_min_confidence: float = 0.5  # MediaPipe Face Detector default
    ally_mood_window_min: float = 30.0  # fusion window, same as the inactivity T
    ally_mood_low_threshold: float = -0.3  # fused valence at/below → low
    ally_mood_positive_threshold: float = 0.3  # fused valence at/above → positive
    ally_mood_weight_face: float = 1.0  # equal weights = unweighted late-fusion baseline (S8 ablation)
    ally_mood_weight_speech: float = 1.0
    ally_mood_weight_movement: float = 1.0
    ally_movement_baseline_days: int = 7  # movement channel absent until a baseline exists
    ally_companion_cooldown_min: float = 120.0  # between mood-triggered COMPANION_CHATs
    ally_checkin_time: str = "10:00"  # scheduled daily check-in, HH:MM local

    # ---- environment (ADR-014/017/018) — SYNTHESIS values, same decisions.md entry -------------
    ally_object_sample_s: float = 10.0  # seconds between detector runs
    ally_sound_min_confidence: float = 0.5  # sigmoid multi-label default threshold
    ally_hazard_sound_labels: tuple[str, ...] = ("Smoke detector, smoke alarm", "Fire alarm", "Shatter")
    ally_logged_sound_labels: tuple[str, ...] = (
        "Screaming",
        "Yell",
        "Crying, sobbing",
        "Glass",
        "Alarm",
        "Alarm clock",
        "Buzzer",
        "Siren",
    )
    ally_fall_evidence_sound_labels: tuple[str, ...] = ("Thump, thud",)
    ally_hazard_cooldown_min: float = 5.0  # same as the stand-down cooldown
    ally_scene_caption_enabled: bool = False  # FR-21: turned on only if the S1 VRAM table allows (P-55)
    ally_caption_sample_s: float = 1800.0  # room caption every 30 min, plus one per event

    # ---- dashboard (hard rule 7) ----------------------------------------------------------------
    ally_dashboard_bind: str = "127.0.0.1"
    ally_dashboard_port: int = 8000
    ally_dashboard_token: SecretStr | None = None

    # ---- paths (git-ignored) ----------------------------------------------------------------------
    ally_data_dir: Path = REPO_ROOT / "data"
    ally_models_dir: Path = REPO_ROOT / "models"
    ally_results_dir: Path = REPO_ROOT / "results"
    ally_db_path: Path = REPO_ROOT / "data" / "ally.sqlite"

    @property
    def sessions_dir(self) -> Path:
        return self.ally_data_dir / "sessions"

    @property
    def keyframes_dir(self) -> Path:
        return self.ally_data_dir / "keyframes"


def get_settings() -> Settings:
    """Construct settings once per process; call sites should receive it, not import a global."""
    return Settings()
