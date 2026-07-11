from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class DecisionStatus(StrEnum):
    ALLOWED = "ALLOWED"
    DENIED = "DENIED"
    REVIEW = "REVIEW"


class FuelingOutcome(StrEnum):
    FUELED = "FUELED"
    NOT_FUELED = "NOT_FUELED"


class DecisionMode(StrEnum):
    MANUAL = "MANUAL"
    AUTO = "AUTO"


@dataclass(frozen=True, slots=True)
class PlateCandidate:
    raw_text: str
    canonical_text: str
    normalized_plate: str | None
    ocr_confidence: float
    detection_confidence: float
    bounding_box: tuple[int, int, int, int]
    validation_error: str | None = None

    @property
    def stable_key(self) -> str:
        return self.normalized_plate or self.canonical_text


@dataclass(frozen=True, slots=True)
class ConfirmedRecognition:
    camera_id: str
    candidate: PlateCandidate
    observed_at: datetime
    average_ocr_confidence: float
    frame: Any
    review_reason: str | None = None


@dataclass(frozen=True, slots=True)
class AccessDecision:
    status: DecisionStatus
    reason: str
    next_allowed_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class RecognitionEvent:
    camera_id: str
    observed_at: datetime
    raw_text: str
    normalized_plate: str | None
    ocr_confidence: float
    detection_confidence: float
    decision: DecisionStatus
    reason: str
    image_path: str | None
