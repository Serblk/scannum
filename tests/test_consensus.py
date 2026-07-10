from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from plate_guard.consensus import TemporalConsensus
from plate_guard.models import PlateCandidate


def candidate(confidence: float = 0.9) -> PlateCandidate:
    return PlateCandidate(
        raw_text="A030BC77",
        canonical_text="А030ВС77",
        normalized_plate="А030ВС77",
        ocr_confidence=confidence,
        detection_confidence=0.95,
        bounding_box=(1, 2, 100, 40),
    )


class TemporalConsensusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.consensus = TemporalConsensus(
            confirmations_required=3,
            window=timedelta(seconds=3),
            duplicate_cooldown=timedelta(seconds=30),
            minimum_ocr_confidence=0.65,
        )
        self.started_at = datetime(2026, 7, 10, 12, tzinfo=UTC)

    def test_emits_after_three_matching_frames(self) -> None:
        self.assertIsNone(
            self.consensus.observe("camera-1", candidate(0.7), self.started_at, "frame-1")
        )
        self.assertIsNone(
            self.consensus.observe(
                "camera-1", candidate(0.8), self.started_at + timedelta(seconds=1), "frame-2"
            )
        )
        result = self.consensus.observe(
            "camera-1", candidate(0.95), self.started_at + timedelta(seconds=2), "frame-3"
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.frame, "frame-3")
        self.assertAlmostEqual(result.average_ocr_confidence, (0.7 + 0.8 + 0.95) / 3)

    def test_duplicate_is_suppressed_across_cameras(self) -> None:
        for index in range(3):
            first = self.consensus.observe(
                "camera-1",
                candidate(),
                self.started_at + timedelta(milliseconds=index * 100),
                index,
            )
        self.assertIsNotNone(first)

        for index in range(3):
            duplicate = self.consensus.observe(
                "camera-2",
                candidate(),
                self.started_at + timedelta(seconds=5, milliseconds=index * 100),
                index,
            )
        self.assertIsNone(duplicate)

    def test_old_observations_do_not_count(self) -> None:
        self.consensus.observe("camera-1", candidate(), self.started_at, "old")
        self.consensus.observe(
            "camera-1", candidate(), self.started_at + timedelta(seconds=4), "new-1"
        )
        result = self.consensus.observe(
            "camera-1", candidate(), self.started_at + timedelta(seconds=5), "new-2"
        )
        self.assertIsNone(result)

    def test_low_confidence_is_ignored(self) -> None:
        for index in range(5):
            result = self.consensus.observe(
                "camera-1",
                candidate(0.4),
                self.started_at + timedelta(milliseconds=index * 100),
                index,
            )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
