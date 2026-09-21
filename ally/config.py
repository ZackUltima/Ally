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
    ally_verify_model: str | None = None
    ally_dialogue_effort: Literal["low", "medium", "high"] = "high"  # re-measured in Sprint 4 (NFR-1)

    # ---- privacy switches (NFR-2, hard rule 3) ------------------------------------------------
    ally_text_only_verify: bool = False
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
    ally_mood_sample_s: float = 5.0

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
