"""Typed messages exchanged between Ally's modules.

Source of truth: docs/context/architecture.md "Module contracts". Every module imports only this file and
its own package; nothing reaches across a boundary to a concrete class. Changing a field here is an
architecture decision (software-architect → ADR in docs/context/decisions.md).

Boundaries
    perception  → fusion   KeypointFrame
    affect      → fusion   AffectSample                              (ADR-013/016, proposed — P-51)
    environment → fusion   SceneObjects, SoundEvent, SceneCaption    (ADR-014/016, proposed — P-51)
    fusion      → agent    PersonState, Event
    agent       → fusion   EscalationRequest, StandDownRequest, ReplyLabel, SpeechSentiment
                           (requests and labels — the engine decides)
    fusion      → notify   Notification   (only event_engine may create one)

No message carries pixels: frames and face crops stay in RAM inside perception/affect/environment (ADR-015).
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

# --------------------------------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------------------------------


class EngineState(StrEnum):
    """States of the event engine FSM (architecture.md §5.1, ADR-001)."""

    NORMAL = "NORMAL"
    SUSPECTED_FALL = "SUSPECTED_FALL"
    SUSPECTED_INACTIVE = "SUSPECTED_INACTIVE"
    VERIFY = "VERIFY"
    CONFIRMED = "CONFIRMED"
    STOOD_DOWN = "STOOD_DOWN"
    FALSE_ALARM = "FALSE_ALARM"
    COMPANION_CHAT = "COMPANION_CHAT"
    SUSPECTED_HAZARD = "SUSPECTED_HAZARD"  # ADR-017: alarm-type sound → spoken check-in


class EventType(StrEnum):
    """Events the engine emits. Each corresponds to entering a non-NORMAL state, plus SUPPRESSED for a
    SUSPECTED_* trigger that failed the debounce (logged, never escalated) and FALL_RECOVERED for a fall the
    person got up from (the guardian is notified; ADR-006). Values are append-only."""

    SUSPECTED_FALL = "SUSPECTED_FALL"
    SUSPECTED_INACTIVE = "SUSPECTED_INACTIVE"
    SUPPRESSED = "SUPPRESSED"
    VERIFY = "VERIFY"
    CONFIRMED = "CONFIRMED"
    STOOD_DOWN = "STOOD_DOWN"
    FALSE_ALARM = "FALSE_ALARM"
    COMPANION_CHAT = "COMPANION_CHAT"
    FALL_RECOVERED = "FALL_RECOVERED"
    SUSPECTED_HAZARD = "SUSPECTED_HAZARD"  # ADR-017


class Posture(StrEnum):
    UPRIGHT = "upright"
    SITTING = "sitting"
    LYING = "lying"
    ON_FLOOR = "on_floor"
    UNKNOWN = "unknown"


Urgency = Literal["low", "medium", "high"]
ReplyLabelValue = Literal["fine", "help", "unclear"]
SentimentValue = Literal["positive", "neutral", "negative"]
Zone = Literal["seat", "bed", "none"]


class MoodLabel(StrEnum):
    """Fused mood estimate (ADR-013): *expressed* mood, never a clinical label (hard rule 6)."""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    LOW = "low"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------------------------------
# perception → fusion
# --------------------------------------------------------------------------------------------------


class KeypointFrame(BaseModel):
    """One frame's worth of perception output. No pixels — keypoints only (hard rule 2)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ts: float = Field(description="monotonic seconds; replay supplies recorded values")
    keypoints: np.ndarray = Field(description="[N, 3] float32 (x_px, y_px, conf); N=17 COCO or 33 MediaPipe")
    bbox: tuple[float, float, float, float] | None = Field(default=None, description="x1, y1, x2, y2 px")
    person_present: bool

    @field_validator("keypoints", mode="before")
    @classmethod
    def _to_array(cls, v: object) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"keypoints must have shape [N, 3], got {arr.shape}")
        return arr

    @field_serializer("keypoints")
    def _ser_array(self, v: np.ndarray) -> list[list[float]]:
        return v.tolist()


# --------------------------------------------------------------------------------------------------
# affect → fusion   (ADR-013/016 — proposed, software-architect review P-51)
# --------------------------------------------------------------------------------------------------


class AffectSample(BaseModel):
    """One facial-expression reading. Numbers only — never pixels, never an identity (ADR-015).

    A face that fails the quality gate gives face_visible=False and valence=None: no sample, never a guess.
    """

    ts: float
    face_visible: bool
    valence: float | None = Field(default=None, ge=-1.0, le=1.0)
    class_probs: dict[str, float] = Field(default_factory=dict, description="FER class → probability")
    quality: float = Field(default=0.0, ge=0.0, le=1.0, description="gate score: size, frontality, conf")

    @field_validator("valence")
    @classmethod
    def _no_valence_without_face(cls, v: float | None, info: object) -> float | None:
        data = getattr(info, "data", {})
        if v is not None and data.get("face_visible") is False:
            raise ValueError("valence requires face_visible=True")
        return v


# --------------------------------------------------------------------------------------------------
# environment → fusion   (ADR-014/016 — proposed, software-architect review P-51)
# --------------------------------------------------------------------------------------------------


class DetectedObject(BaseModel):
    label: str = Field(min_length=1, description="detector class name, e.g. COCO 'couch'")
    bbox: tuple[float, float, float, float] = Field(description="x1, y1, x2, y2 px")
    confidence: float = Field(ge=0.0, le=1.0)
    on_floor: bool = False


class SceneObjects(BaseModel):
    ts: float
    objects: list[DetectedObject] = Field(default_factory=list)


class SoundEvent(BaseModel):
    """An allow-listed household sound (alarm-type, glass breaking, thud, …). Audio itself stays in RAM."""

    ts: float
    label: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class SceneCaption(BaseModel):
    """Local VLM caption (FR-21). Stays on the laptop, never goes to the cloud LLM (ADR-018)."""

    ts: float
    text: str = Field(min_length=1, max_length=500)


# --------------------------------------------------------------------------------------------------
# fusion → agent
# --------------------------------------------------------------------------------------------------


class PersonState(BaseModel):
    """Compact JSON summary of the person; this is what crosses the privacy boundary to the LLM."""

    posture: Posture = Posture.UNKNOWN
    motion_energy: float = Field(default=0.0, ge=0.0)
    mood: float | None = Field(
        default=None, ge=-1.0, le=1.0, description="fused valence from fusion/mood.py (ADR-013)"
    )
    mood_label: MoodLabel = MoodLabel.UNKNOWN
    activity_level: float | None = Field(
        default=None, ge=0.0, description="activity vs the person's baseline (FR-22); None = no baseline yet"
    )
    zone: Zone | None = Field(default=None, description="seat / bed / none (ADR-011); None = no zone map")
    presence: bool = False
    inactive_for_s: float = Field(default=0.0, ge=0.0)


class Event(BaseModel):
    type: EventType
    ts: float
    confidence: float = Field(ge=0.0, le=1.0)
    keyframe_id: str | None = None
    state: PersonState


# --------------------------------------------------------------------------------------------------
# agent → fusion  (requests; ally/fusion/event_engine.py approves or rejects — ADR-001)
# --------------------------------------------------------------------------------------------------


class EscalationRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=500)
    urgency: Urgency = "medium"


class StandDownRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=500)
    urgency: Urgency = "low"


class ReplyLabel(BaseModel):
    """Structured output of the reply classifier. The FSM consumes `label`; the model never transitions."""

    label: ReplyLabelValue
    transcript: str = ""


class SpeechSentiment(BaseModel):
    """Speech-text mood channel (FR-4, ADR-013): what the person said, labelled by the LLM. Never an alert."""

    label: SentimentValue
    transcript: str = ""


# --------------------------------------------------------------------------------------------------
# fusion → notify
# --------------------------------------------------------------------------------------------------


class Notification(BaseModel):
    """Only ally/fusion/event_engine.py may construct one (hard rule 1)."""

    event: Event
    keyframe_path: Path | None = None
    transcript: str = ""
    blur_face: bool = True


__all__ = [
    "AffectSample",
    "DetectedObject",
    "EngineState",
    "EscalationRequest",
    "Event",
    "EventType",
    "KeypointFrame",
    "MoodLabel",
    "Notification",
    "PersonState",
    "Posture",
    "ReplyLabel",
    "ReplyLabelValue",
    "SceneCaption",
    "SceneObjects",
    "SentimentValue",
    "SoundEvent",
    "SpeechSentiment",
    "StandDownRequest",
    "Urgency",
    "Zone",
]
