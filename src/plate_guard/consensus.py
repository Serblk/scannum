from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import mean
from typing import Any, Deque

from .models import ConfirmedRecognition, PlateCandidate


@dataclass(frozen=True, slots=True)
class _Observation:
    candidate: PlateCandidate
    observed_at: datetime
    frame: Any


class TemporalConsensus:
    """Подтверждает одинаковый OCR-результат несколькими близкими по времени кадрами."""

    def __init__(
        self,
        confirmations_required: int,
        window: timedelta,
        duplicate_cooldown: timedelta,
        minimum_ocr_confidence: float,
    ) -> None:
        if confirmations_required <= 0:
            raise ValueError("Число подтверждений должно быть положительным")
        if window <= timedelta(0) or duplicate_cooldown <= timedelta(0):
            raise ValueError("Временные интервалы должны быть положительными")
        if not 0 < minimum_ocr_confidence <= 1:
            raise ValueError("Порог OCR должен находиться в диапазоне (0, 1]")
        self._confirmations_required = confirmations_required
        self._window = window
        self._duplicate_cooldown = duplicate_cooldown
        self._minimum_ocr_confidence = minimum_ocr_confidence
        self._observations: dict[tuple[str, str], Deque[_Observation]] = defaultdict(deque)
        self._last_emitted: dict[str, datetime] = {}

    def observe(
        self,
        camera_id: str,
        candidate: PlateCandidate,
        observed_at: datetime,
        frame: Any,
    ) -> ConfirmedRecognition | None:
        _require_aware(observed_at)
        if candidate.ocr_confidence < self._minimum_ocr_confidence:
            return None
        if not candidate.stable_key:
            return None

        key = (camera_id, candidate.stable_key)
        observations = self._observations[key]
        cutoff = observed_at - self._window
        while observations and observations[0].observed_at < cutoff:
            observations.popleft()
        observations.append(_Observation(candidate, observed_at, _copy_frame(frame)))

        if len(observations) < self._confirmations_required:
            return None

        last_emitted = self._last_emitted.get(candidate.stable_key)
        if last_emitted is not None and observed_at - last_emitted < self._duplicate_cooldown:
            observations.clear()
            return None

        relevant = list(observations)[-self._confirmations_required :]
        best = max(relevant, key=lambda item: item.candidate.ocr_confidence)
        average_confidence = mean(item.candidate.ocr_confidence for item in relevant)
        self._last_emitted[candidate.stable_key] = observed_at
        self._clear_plate(candidate.stable_key)
        self._prune_old_emissions(observed_at)
        return ConfirmedRecognition(
            camera_id=camera_id,
            candidate=best.candidate,
            observed_at=observed_at,
            average_ocr_confidence=average_confidence,
            frame=best.frame,
        )

    def _clear_plate(self, stable_key: str) -> None:
        for key in [key for key in self._observations if key[1] == stable_key]:
            del self._observations[key]

    def _prune_old_emissions(self, now: datetime) -> None:
        cutoff = now - self._duplicate_cooldown * 2
        for plate in [plate for plate, value in self._last_emitted.items() if value < cutoff]:
            del self._last_emitted[plate]


def _copy_frame(frame: Any) -> Any:
    copy = getattr(frame, "copy", None)
    return copy() if callable(copy) else frame


def _require_aware(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Время наблюдения должно содержать часовой пояс")
