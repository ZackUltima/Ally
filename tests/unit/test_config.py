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


def test_adr_005_006_defaults_match_decisions(clean_env):
    # decisions.md "Proposed values for the ADR-005/006 settings", accepted provisionally (P-39, 23 Sep 2026)
    s = Settings(_env_file=None)
    assert s.ally_decision_posture_window_s == 2.0
    assert s.ally_recovery_min_down_s is None  # every impact that reaches the floor counts


def test_new_channel_numbers_match_decisions(clean_env):
    # decisions.md "Numbers for ADR-013 / ADR-014 / ADR-017 (P-54)", SYNTHESIS, 25 Sep 2026
    s = Settings(_env_file=None)
    assert s.ally_affect_sample_s == 1.0
    assert (s.ally_face_min_px, s.ally_face_max_yaw_deg, s.ally_face_min_confidence) == (48, 30.0, 0.5)
    assert s.ally_mood_window_min == s.ally_inactivity_minutes == 30.0
    assert (s.ally_mood_low_threshold, s.ally_mood_positive_threshold) == (-0.3, 0.3)
    assert s.ally_mood_weight_face == s.ally_mood_weight_speech == s.ally_mood_weight_movement == 1.0
    assert s.ally_movement_baseline_days == 7
    assert s.ally_companion_cooldown_min == 120.0
    assert s.ally_checkin_time == "10:00"
    assert s.ally_object_sample_s == 10.0
    assert s.ally_sound_min_confidence == 0.5
    assert s.ally_hazard_cooldown_min == s.ally_stand_down_cooldown_min == 5.0
    assert s.ally_caption_sample_s == 1800.0
    assert s.ally_scene_caption_enabled is False  # FR-21 only if the S1 VRAM table allows (P-55)


def test_hazard_sound_list_is_narrow(clean_env):
    # ADR-017: only these trigger a check-in; TV-prone and generic labels are logged, never trigger
    s = Settings(_env_file=None)
    assert s.ally_hazard_sound_labels == ("Smoke detector, smoke alarm", "Fire alarm", "Shatter")
    assert not set(s.ally_hazard_sound_labels) & set(s.ally_logged_sound_labels)
    assert "Screaming" in s.ally_logged_sound_labels and "Alarm clock" in s.ally_logged_sound_labels
    assert s.ally_fall_evidence_sound_labels == ("Thump, thud",)


def test_privacy_and_network_defaults(clean_env):
    s = Settings(_env_file=None)
    assert s.ally_guardian_keyframe is True
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
    monkeypatch.setenv("ALLY_GUARDIAN_KEYFRAME", "false")
    monkeypatch.setenv("ALLY_INACTIVITY_MINUTES", "45")
    s = Settings(_env_file=None)
    assert s.ally_guardian_keyframe is False
    assert s.ally_inactivity_minutes == 45
