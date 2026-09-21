"""Defaults match the numbers fixed in docs/context/ (hard rule 5)."""

from __future__ import annotations

from ally.config import Settings


def test_defaults_match_documented_numbers(clean_env):
    s = Settings(_env_file=None)
    assert (s.ally_capture_width, s.ally_capture_height) == (640, 480)  # hard rule 8
    assert s.ally_target_fps == 15
    assert s.ally_inactivity_minutes == 30
    assert s.ally_keyframe_retention_days == 7
    assert s.ally_debounce_s == 3.0
    assert s.ally_verify_max_prompts == 2
    assert s.ally_verify_listen_s == 20.0
    assert s.ally_stand_down_cooldown_min == 5.0
    assert s.ally_fall_threshold is None  # τ_fall is chosen in Sprint 2, not invented here


def test_privacy_and_network_defaults(clean_env):
    s = Settings(_env_file=None)
    assert s.ally_text_only_verify is False
    assert s.ally_blur_face_for_guardian is True
    assert s.ally_dashboard_bind == "127.0.0.1"  # never 0.0.0.0 (hard rule 7)


def test_secrets_are_not_printable(clean_env, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-real-key")
    s = Settings(_env_file=None)
    assert "sk-test" not in repr(s)
    assert "sk-test" not in str(s)
    assert s.anthropic_api_key is not None
    assert s.anthropic_api_key.get_secret_value() == "sk-test-not-a-real-key"


def test_env_overrides(clean_env, monkeypatch):
    monkeypatch.setenv("ALLY_TEXT_ONLY_VERIFY", "true")
    monkeypatch.setenv("ALLY_INACTIVITY_MINUTES", "45")
    s = Settings(_env_file=None)
    assert s.ally_text_only_verify is True
    assert s.ally_inactivity_minutes == 45
