"""Typed messages exchanged between Ally's modules.

Source of truth: docs/context/architecture.md "Module contracts". Every module imports only this file and
its own package; nothing reaches across a boundary to a concrete class. Changing a field here is an
architecture decision (software-architect → ADR in docs/context/decisions.md).

Boundaries
    perception → fusion   KeypointFrame
    fusion     → agent    PersonState, Event
    agent      → fusion   EscalationRequest, StandDownRequest, ReplyLabel (requests — the engine decides)
    llm_client → fusion   VerifyResult   (a second vote, never a decision)
    fusion     → notify   Notification   (only event_engine may create one)
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


class EventType(StrEnum):
    """Events the engine emits. Each corresponds to entering a non-NORMAL state, plus SUPPRESSED for a
    SUSPECTED_* trigger that failed the debounce (logged, never escalated)."""

    SUSPECTED_FALL = "SUSPECTED_FALL"
    SUSPECTED_INACTIVE = "SUSPECTED_INACTIVE"
    SUPPRESSED = "SUPPRESSED"
    VERIFY = "VERIFY"
    CONFIRMED = "CONFIRMED"
    STOOD_DOWN = "STOOD_DOWN"
    FALSE_ALARM = "FALSE_ALARM"
    COMPANION_CHAT = "COMPANION_CHAT"


class Posture(StrEnum):
    UPRIGHT = "upright"
    SITTING = "sitting"
    LYING = "lying"
    ON_FLOOR = "on_floor"
    UNKNOWN = "unknown"


Urgency = Literal["low", "medium", "high"]
ReplyLabelValue = Literal["fine", "help", "unclear"]


# --------------------------------------------------------------------------------------------------
# perception → fusion
# --------------------------------------------------------------------------------------------------


class KeypointFrame(BaseModel):
    """One frame's worth of perception output. No pixels — keypoints only (hard rule 2)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ts: float = Field(description="monotonic seconds; replay supplies recorded values")
    keypoints: np.ndarray = Field(description="[N, 3] float32 (x_px, y_px, conf); N=17 COCO or 33 MediaPipe")
    bbox: tuple[float, float, float, float] | None = Field(default=None, description="x1, y1, x2, y2 px")
    face_valence: float | None = Field(default=None, ge=-1.0, le=1.0)
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
# fusion → agent
# --------------------------------------------------------------------------------------------------


class PersonState(BaseModel):
    """Compact JSON summary of the person; this is what crosses the privacy boundary to the LLM."""

    posture: Posture = Posture.UNKNOWN
    motion_energy: float = Field(default=0.0, ge=0.0)
    mood: float | None = Field(default=None, ge=-1.0, le=1.0, description="EMA valence; None if no face")
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


# --------------------------------------------------------------------------------------------------
# llm_client → fusion
# --------------------------------------------------------------------------------------------------


class VerifyResult(BaseModel):
    """Structured answer from the VLM keyframe check — a second vote alongside posture (FR-5)."""

    person_on_floor: bool
    confidence: float = Field(ge=0.0, le=1.0)
    hazard: str | None = None
    summary: str = Field(default="", max_length=300, description="one sentence for the guardian")


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
    "EngineState",
    "EscalationRequest",
    "Event",
    "EventType",
    "KeypointFrame",
    "Notification",
    "PersonState",
    "Posture",
    "ReplyLabel",
    "ReplyLabelValue",
    "StandDownRequest",
    "Urgency",
    "VerifyResult",
]
