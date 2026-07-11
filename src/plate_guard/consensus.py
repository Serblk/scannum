from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import mean
from typing import Any, Sequence

from .models import ConfirmedRecognition, PlateCandidate


@dataclass(frozen=True, slots=True)
class _Observation:
    candidate: PlateCandidate
    observed_at: datetime
    frame: Any


@dataclass(slots=True)
class _Track:
    camera_id: str
    started_at: datetime
    last_seen_at: datetime
    bounding_box: tuple[int, int, int, int]
    observations: list[_Observation] = field(default_factory=list)


class TemporalConsensus:
    """Собирает OCR-варианты одной физической рамки и выдаёт один итог."""

    def __init__(
        self,
        confirmations_required: int,
        window: timedelta,
        duplicate_cooldown: timedelta,
        minimum_ocr_confidence: float,
        winner_share: float = 0.70,
    ) -> None:
        if confirmations_required <= 0:
            raise ValueError("Число подтверждений должно быть положительным")
        if window <= timedelta(0) or duplicate_cooldown <= timedelta(0):
            raise ValueError("Временные интервалы должны быть положительными")
        if not 0 < minimum_ocr_confidence <= 1 or not 0.5 < winner_share <= 1:
            raise ValueError("Пороги уверенности должны находиться в диапазоне (0, 1]")
        self._confirmations_required = confirmations_required
        self._window = window
        self._duplicate_cooldown = duplicate_cooldown
        self._minimum_ocr_confidence = minimum_ocr_confidence
        self._winner_share = winner_share
        self._tracks: list[_Track] = []
        self._last_emitted: dict[str, datetime] = {}

    def observe_many(
        self,
        camera_id: str,
        candidates: Sequence[PlateCandidate],
        observed_at: datetime,
        frame: Any,
    ) -> list[ConfirmedRecognition]:
        _require_aware(observed_at)
        used_tracks: set[int] = set()
        valid_candidates = [
            candidate
            for candidate in candidates
            if candidate.normalized_plate is not None
            and candidate.ocr_confidence >= self._minimum_ocr_confidence
        ]
        for candidate in valid_candidates:
            track = self._matching_track(camera_id, candidate.bounding_box, used_tracks)
            if track is None:
                track = _Track(
                    camera_id=camera_id,
                    started_at=observed_at,
                    last_seen_at=observed_at,
                    bounding_box=candidate.bounding_box,
                )
                self._tracks.append(track)
            track.last_seen_at = observed_at
            track.bounding_box = candidate.bounding_box
            track.observations.append(
                _Observation(candidate, observed_at, _copy_frame(frame))
            )
            used_tracks.add(id(track))

        ready: list[ConfirmedRecognition] = []
        remaining: list[_Track] = []
        for track in self._tracks:
            old_without_enough_data = (
                observed_at - track.last_seen_at > self._window
                and len(track.observations) < self._confirmations_required
            )
            if old_without_enough_data:
                continue
            if observed_at - track.started_at < self._window:
                remaining.append(track)
                continue
            if len(track.observations) < self._confirmations_required:
                remaining.append(track)
                continue
            result = self._resolve_track(track, observed_at)
            if result is not None:
                ready.append(result)
        self._tracks = remaining
        self._prune_old_emissions(observed_at)
        return ready

    def observe(
        self,
        camera_id: str,
        candidate: PlateCandidate,
        observed_at: datetime,
        frame: Any,
    ) -> ConfirmedRecognition | None:
        results = self.observe_many(camera_id, [candidate], observed_at, frame)
        return results[0] if results else None

    def _matching_track(
        self,
        camera_id: str,
        bounding_box: tuple[int, int, int, int],
        used_tracks: set[int],
    ) -> _Track | None:
        matches = [
            track
            for track in self._tracks
            if track.camera_id == camera_id
            and id(track) not in used_tracks
            and _intersection_over_union(track.bounding_box, bounding_box) >= 0.30
        ]
        return max(
            matches,
            key=lambda track: _intersection_over_union(track.bounding_box, bounding_box),
            default=None,
        )

    def _resolve_track(
        self, track: _Track, observed_at: datetime
    ) -> ConfirmedRecognition | None:
        if len(track.observations) < self._confirmations_required:
            return None
        counts = Counter(
            observation.candidate.normalized_plate
            for observation in track.observations
            if observation.candidate.normalized_plate is not None
        )
        counts = _merge_truncated_regions(counts)
        winner, winner_count = counts.most_common(1)[0]
        total = sum(counts.values())
        variants = counts.most_common()
        ambiguous = len(variants) > 1 and winner_count / total < self._winner_share
        emission_key = "|".join(sorted(counts)) if ambiguous else winner
        last_emitted = self._last_emitted.get(emission_key)
        if last_emitted is not None and observed_at - last_emitted < self._duplicate_cooldown:
            return None

        winner_observations = [
            item
            for item in track.observations
            if _supports_winner(item.candidate.normalized_plate, winner, counts)
        ]
        exact_winner_observations = [
            item
            for item in winner_observations
            if item.candidate.normalized_plate == winner
        ]
        best = max(
            exact_winner_observations or winner_observations,
            key=lambda item: item.candidate.ocr_confidence,
        )
        average_confidence = mean(
            item.candidate.ocr_confidence for item in winner_observations
        )
        review_reason = _ambiguity_reason(variants, total) if ambiguous else None
        self._last_emitted[emission_key] = observed_at
        return ConfirmedRecognition(
            camera_id=track.camera_id,
            candidate=best.candidate,
            observed_at=observed_at,
            average_ocr_confidence=average_confidence,
            frame=best.frame,
            review_reason=review_reason,
        )

    def _prune_old_emissions(self, now: datetime) -> None:
        cutoff = now - self._duplicate_cooldown * 2
        self._last_emitted = {
            key: value for key, value in self._last_emitted.items() if value >= cutoff
        }


def _merge_truncated_regions(counts: Counter[str]) -> Counter[str]:
    merged = counts.copy()
    plates = list(counts)
    for short in plates:
        if len(short) != 8 or short not in merged:
            continue
        full_matches = [full for full in plates if len(full) == 9 and full.startswith(short)]
        if not full_matches:
            continue
        strongest = max(full_matches, key=lambda value: counts[value])
        if counts[strongest] >= 2 and counts[strongest] >= counts[short]:
            merged[strongest] += merged.pop(short)
    return merged


def _supports_winner(plate: str | None, winner: str, counts: Counter[str]) -> bool:
    if plate == winner:
        return True
    return (
        plate is not None
        and len(plate) == 8
        and len(winner) == 9
        and winner.startswith(plate)
        and plate not in counts
    )


def _ambiguity_reason(variants: list[tuple[str, int]], total: int) -> str:
    lines = ["OCR не смог однозначно определить номер:"]
    lines.extend(f"{plate} — {count / total:.0%}" for plate, count in variants[:3])
    return "\n".join(lines)


def _intersection_over_union(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _copy_frame(frame: Any) -> Any:
    copy = getattr(frame, "copy", None)
    return copy() if callable(copy) else frame


def _require_aware(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Время наблюдения должно содержать часовой пояс")
