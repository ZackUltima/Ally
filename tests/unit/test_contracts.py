"""Boundary types behave as architecture.md promises."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from ally.contracts import (
    EngineState,
    EscalationRequest,
    Event,
    EventType,
    KeypointFrame,
    Notification,
    PersonState,
    Posture,
    ReplyLabel,
    StandDownRequest,
    VerifyResult,
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
    kf = KeypointFrame(
        ts=2.5, keypoints=np.ones((33, 3)), bbox=(0, 0, 10, 10), face_valence=-0.2, person_present=True
    )
    again = KeypointFrame.model_validate_json(kf.model_dump_json())
    assert np.array_equal(again.keypoints, kf.keypoints)
    assert again.bbox == kf.bbox
    assert again.face_valence == kf.face_valence


def test_face_valence_is_bounded():
    with pytest.raises(ValidationError):
        KeypointFrame(ts=0.0, keypoints=np.zeros((17, 3)), face_valence=1.5, person_present=True)


def test_person_state_defaults_are_unknown_and_absent():
    s = PersonState()
    assert s.posture is Posture.UNKNOWN
    assert s.presence is False
    assert s.mood is None


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


def test_verify_result_and_notification_compose():
    ev = Event(type=EventType.CONFIRMED, ts=10.0, confidence=0.93, keyframe_id="kf-1", state=PersonState())
    vr = VerifyResult(person_on_floor=True, confidence=0.8, summary="Person lying on the floor.")
    n = Notification(event=ev, transcript="(no reply)", blur_face=True)
    assert vr.person_on_floor
    assert n.event.type is EventType.CONFIRMED
    assert n.keyframe_path is None


def test_every_non_normal_state_has_an_event_type():
    # Scaffold-era assumption (Sprint 0): each non-NORMAL state is announced by an event of the same name.
    # If the Sprint 3 FSM design adds a state that emits no event, drop this test in the same ADR.
    event_names = {e.value for e in EventType}
    for st in EngineState:
        if st is EngineState.NORMAL:
            continue
        assert st.value in event_names
