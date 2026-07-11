from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from plate_guard.consensus import TemporalConsensus
from plate_guard.models import PlateCandidate


def candidate(
    plate: str = "А030ВС77",
    confidence: float = 0.9,
    box: tuple[int, int, int, int] = (1, 2, 100, 40),
) -> PlateCandidate:
    return PlateCandidate(
        raw_text=plate,
        canonical_text=plate,
        normalized_plate=plate,
        ocr_confidence=confidence,
        detection_confidence=0.95,
        bounding_box=box,
    )


class TemporalConsensusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.consensus = TemporalConsensus(
            confirmations_required=4,
            window=timedelta(seconds=3),
            duplicate_cooldown=timedelta(seconds=30),
            minimum_ocr_confidence=0.70,
        )
        self.started_at = datetime(2026, 7, 10, 12, tzinfo=UTC)

    def test_waits_for_window_and_emits_stable_plate(self) -> None:
        result = []
        for index in range(4):
            result = self.consensus.observe_many(
                "camera-1",
                [candidate(confidence=0.75 + index * 0.05)],
                self.started_at + timedelta(seconds=index),
                f"frame-{index}",
            )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].candidate.normalized_plate, "А030ВС77")
        self.assertIsNone(result[0].review_reason)

    def test_truncated_region_is_merged_into_stable_full_region(self) -> None:
        variants = ["С888КТ19", "С888КТ197", "С888КТ197", "С888КТ197"]
        result = []
        for index, plate in enumerate(variants):
            result = self.consensus.observe_many(
                "camera-1",
                [candidate(plate)],
                self.started_at + timedelta(seconds=index),
                f"frame-{index}",
            )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].candidate.normalized_plate, "С888КТ197")
        self.assertIsNone(result[0].review_reason)

    def test_close_competing_plate_texts_require_review(self) -> None:
        variants = ["С888КТ197", "С888КГ197", "С888КТ197", "С888КГ197"]
        result = []
        for index, plate in enumerate(variants):
            result = self.consensus.observe_many(
                "camera-1",
                [candidate(plate)],
                self.started_at + timedelta(seconds=index),
                f"frame-{index}",
            )
        self.assertEqual(len(result), 1)
        self.assertIsNotNone(result[0].review_reason)
        assert result[0].review_reason is not None
        self.assertIn("С888КТ197", result[0].review_reason)
        self.assertIn("С888КГ197", result[0].review_reason)

    def test_different_locations_are_independent_tracks(self) -> None:
        result = []
        for index in range(4):
            result = self.consensus.observe_many(
                "camera-1",
                [
                    candidate("А111АА77", box=(0, 0, 100, 40)),
                    candidate("В222ВВ77", box=(300, 0, 400, 40)),
                ],
                self.started_at + timedelta(seconds=index),
                index,
            )
        self.assertEqual(
            {item.candidate.normalized_plate for item in result},
            {"А111АА77", "В222ВВ77"},
        )

    def test_duplicate_is_suppressed_across_cameras(self) -> None:
        first = []
        for index in range(4):
            first = self.consensus.observe_many(
                "camera-1",
                [candidate()],
                self.started_at + timedelta(seconds=index),
                index,
            )
        self.assertEqual(len(first), 1)
        duplicate = []
        for index in range(4):
            duplicate = self.consensus.observe_many(
                "camera-2",
                [candidate()],
                self.started_at + timedelta(seconds=5 + index),
                index,
            )
        self.assertEqual(duplicate, [])

    def test_low_confidence_and_invalid_plate_are_ignored(self) -> None:
        invalid = candidate("МУСОР", confidence=0.95)
        invalid = PlateCandidate(
            raw_text=invalid.raw_text,
            canonical_text=invalid.canonical_text,
            normalized_plate=None,
            ocr_confidence=invalid.ocr_confidence,
            detection_confidence=invalid.detection_confidence,
            bounding_box=invalid.bounding_box,
        )
        result = []
        for index in range(5):
            result = self.consensus.observe_many(
                "camera-1",
                [candidate(confidence=0.4), invalid],
                self.started_at + timedelta(seconds=index),
                index,
            )
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
