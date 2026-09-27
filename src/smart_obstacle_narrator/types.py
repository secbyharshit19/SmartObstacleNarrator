"""Shared immutable boundaries between detection, decisions, and speech."""
from dataclasses import dataclass

CLASSES = ("person", "chair", "bicycle", "vehicle", "stairs", "pole", "barrier", "door")

@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]

@dataclass(frozen=True)
class Observation:
    detection: Detection
    direction: str
    distance_m: float | None
    distance_reason: str
    distance_band: str
    central_overlap: float

@dataclass(frozen=True)
class WarningEvent:
    event_id: str
    track_id: int
    text: str
    score: float
    created: float
    expires: float
    signature: tuple[str, str]

@dataclass(frozen=True)
class SpeechFeedback:
    event: WarningEvent
    kind: str
    timestamp: float
    detail: str = ""
