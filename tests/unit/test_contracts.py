"""Boundary types behave as architecture.md promises."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from ally.contracts import (
    AffectSample,
    DetectedObject,
    EngineState,
    EscalationRequest,
    Event,
    EventType,
    KeypointFrame,
    MoodLabel,
    Notification,
    PersonState,
    Posture,
    ReplyLabel,
    SceneCaption,
    SceneObjects,
    SoundEvent,
    SpeechSentiment,
    StandDownRequest,
)


def test_keypoint_frame_accepts_list_and_stores_float32_array():
    kf = KeypointFrame(ts=1.0, keypoints=[[1, 2, 0.9]] * 17, person_present=True)
    assert isinstance(kf.keypoints, np.ndarray)
    assert kf.keypoints.shape == (17, 3)
    assert kf.keypoints.dtype == np.float32


@pytest.mark.parametrize("bad", [[[1, 2]], [1, 2, 3], np.zeros((17, 4))])
def test_keypoint_frame_rejects_wrong_shape(bad):
    with pytest.raises(ValidationError):
        KeypointFrame(ts=0.0, keypoints=bad, person_present=True)


def test_keypoint_frame_round_trips_through_json():
    kf = KeypointFrame(ts=2.5, keypoints=np.ones((33, 3)), bbox=(0, 0, 10, 10), person_present=True)
    again = KeypointFrame.model_validate_json(kf.model_dump_json())
    assert np.array_equal(again.keypoints, kf.keypoints)
    assert again.bbox == kf.bbox


def test_keypoint_frame_carries_keypoints_only():
    # ADR-016: facial affect travels in its own AffectSample; the per-frame pose message stays pose-only.
    assert "face_valence" not in KeypointFrame.model_fields


def test_person_state_defaults_are_unknown_and_absent():
    s = PersonState()
    assert s.posture is Posture.UNKNOWN
    assert s.presence is False
    assert s.mood is None
    assert s.mood_label is MoodLabel.UNKNOWN  # no channel yet → unknown, never "neutral" (ADR-013)
    assert s.activity_level is None
    assert s.zone is None


def test_mood_label_is_never_clinical():
    # hard rule 6: the label vocabulary is closed and non-clinical
    assert [m.value for m in MoodLabel] == ["positive", "neutral", "low", "unknown"]


def test_affect_sample_is_numbers_only_and_needs_a_face_for_valence():
    ok = AffectSample(ts=1.0, face_visible=True, valence=-0.4, class_probs={"sad": 0.6}, quality=0.8)
    assert ok.valence == -0.4
    gated = AffectSample(ts=1.0, face_visible=False)
    assert gated.valence is None
    with pytest.raises(ValidationError):
        AffectSample(ts=1.0, face_visible=False, valence=0.2)  # a gated face never yields a guess
    with pytest.raises(ValidationError):
        AffectSample(ts=1.0, face_visible=True, valence=1.5)
    # ADR-015: no field can carry pixels
    assert set(AffectSample.model_fields) == {"ts", "face_visible", "valence", "class_probs", "quality"}


def test_speech_sentiment_is_closed_vocabulary():
    assert SpeechSentiment(label="negative", transcript="a bit lonely today").label == "negative"
    with pytest.raises(ValidationError):
        SpeechSentiment(label="depressed")  # type: ignore[arg-type]


def test_environment_messages_validate():
    objs = SceneObjects(ts=2.0, objects=[DetectedObject(label="couch", bbox=(0, 0, 50, 40), confidence=0.9)])
    assert objs.objects[0].on_floor is False
    with pytest.raises(ValidationError):
        SoundEvent(ts=0.0, label="glass", confidence=1.5)
    with pytest.raises(ValidationError):
        SceneCaption(ts=0.0, text="")


def test_event_requires_confidence_in_unit_interval():
    with pytest.raises(ValidationError):
        Event(type=EventType.SUSPECTED_FALL, ts=0.0, confidence=1.2, state=PersonState())


def test_reply_label_is_closed_vocabulary():
    assert ReplyLabel(label="fine").label == "fine"
    with pytest.raises(ValidationError):
        ReplyLabel(label="ok")  # type: ignore[arg-type]


def test_requests_need_a_reason():
    with pytest.raises(ValidationError):
        EscalationRequest(reason="")
    assert StandDownRequest(reason="person said they are fine").urgency == "low"
    assert EscalationRequest(reason="person asked for help").urgency == "medium"


def test_notification_composes_from_event():
    ev = Event(type=EventType.CONFIRMED, ts=10.0, confidence=0.93, keyframe_id="kf-1", state=PersonState())
    n = Notification(event=ev, transcript="(no reply)", blur_face=True)
    assert n.event.type is EventType.CONFIRMED
    assert n.keyframe_path is None


def test_event_type_values_are_append_only():
    # Enum values are the wire format of expected_events.json and SQLite rows (architecture.md): new values
    # go at the end, existing ones never change. FALL_RECOVERED appended by ADR-006.
    assert [e.value for e in EventType] == [
        "SUSPECTED_FALL",
        "SUSPECTED_INACTIVE",
        "SUPPRESSED",
        "VERIFY",
        "CONFIRMED",
        "STOOD_DOWN",
        "FALSE_ALARM",
        "COMPANION_CHAT",
        "FALL_RECOVERED",
        "SUSPECTED_HAZARD",  # ADR-017
    ]


def test_every_non_normal_state_has_an_event_type():
    # Scaffold-era assumption (Sprint 0): each non-NORMAL state is announced by an event of the same name.
    # If the Sprint 3 FSM design adds a state that emits no event, drop this test in the same ADR.
    event_names = {e.value for e in EventType}
    for st in EngineState:
        if st is EngineState.NORMAL:
            continue
        assert st.value in event_names
