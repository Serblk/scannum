from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from plate_guard.config import CameraConfig
from plate_guard.models import (
    DecisionMode,
    DecisionStatus,
    FuelingOutcome,
    RecognitionEvent,
)
from plate_guard.storage import SQLiteRepository
from plate_guard.rules import evaluate_access


class SQLiteRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        database_path = Path(self.temporary_directory.name) / "system.db"
        self.repository = SQLiteRepository(database_path)
        self.repository.initialize()
        self.repository.upsert_cameras(
            [CameraConfig(id="camera-1", name="Камера 1", source=0)]
        )
        self.observed_at = datetime(2026, 7, 10, 9, tzinfo=UTC)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_records_and_confirms_allowed_fueling(self) -> None:
        recognition_id = self.repository.record_recognition(self._event(DecisionStatus.ALLOWED))
        self.assertGreater(recognition_id, 0)
        self.assertIsNone(self.repository.last_confirmed_fueling("А030ВС77"))

        confirmed_at = datetime(2026, 7, 10, 10, tzinfo=UTC)
        fueling_id = self.repository.confirm_latest_allowed_fueling(
            "А030ВС77", confirmed_at, "Домашний тест"
        )
        self.assertGreater(fueling_id, 0)
        self.assertEqual(self.repository.last_confirmed_fueling("А030ВС77"), confirmed_at)

        rows = self.repository.export_rows()
        self.assertEqual(rows[0]["operator_note"], "Домашний тест")
        self.assertEqual(rows[0]["normalized_plate"], "А030ВС77")

    def test_denied_event_cannot_be_confirmed(self) -> None:
        self.repository.record_recognition(self._event(DecisionStatus.DENIED))
        with self.assertRaises(LookupError):
            self.repository.confirm_latest_allowed_fueling(
                "А030ВС77", datetime(2026, 7, 10, 10, tzinfo=UTC)
            )

    def test_one_recognition_cannot_be_confirmed_twice(self) -> None:
        self.repository.record_recognition(self._event(DecisionStatus.ALLOWED))
        confirmed_at = datetime(2026, 7, 10, 10, tzinfo=UTC)
        self.repository.confirm_latest_allowed_fueling("А030ВС77", confirmed_at)
        with self.assertRaises(LookupError):
            self.repository.confirm_latest_allowed_fueling("А030ВС77", confirmed_at)

    def test_manual_not_fueled_does_not_start_eight_hour_timer(self) -> None:
        recognition_id = self.repository.record_recognition(self._event(DecisionStatus.ALLOWED))
        self.assertEqual(
            self.repository.pending_allowed_recognition("А030ВС77"), recognition_id
        )

        self.repository.resolve_fueling_decision(
            recognition_id=recognition_id,
            outcome=FuelingOutcome.NOT_FUELED,
            mode=DecisionMode.MANUAL,
            decided_at=datetime(2026, 7, 10, 10, tzinfo=UTC),
        )

        self.assertIsNone(self.repository.last_confirmed_fueling("А030ВС77"))
        self.assertIsNone(self.repository.pending_allowed_recognition("А030ВС77"))
        row = self.repository.export_rows()[0]
        self.assertEqual(row["fueling_outcome"], "NOT_FUELED")
        self.assertEqual(row["decision_mode"], "MANUAL")

    def test_auto_fueled_starts_eight_hour_timer(self) -> None:
        confirmed_at = datetime(2026, 7, 10, 10, tzinfo=UTC)
        event = RecognitionEvent(
            camera_id="camera-1",
            observed_at=confirmed_at,
            raw_text="A030BC77",
            normalized_plate="А030ВС77",
            ocr_confidence=0.9,
            detection_confidence=0.95,
            decision=DecisionStatus.ALLOWED,
            reason="Тест",
            image_path=None,
        )
        self.repository.record_recognition(event, auto_confirm=True)

        self.assertEqual(self.repository.last_confirmed_fueling("А030ВС77"), confirmed_at)
        row = self.repository.export_rows()[0]
        self.assertEqual(row["fueling_outcome"], "FUELED")
        self.assertEqual(row["decision_mode"], "AUTO")
        second_attempt = evaluate_access(
            "А030ВС77",
            datetime(2026, 7, 10, 10, 1, tzinfo=UTC),
            self.repository.last_confirmed_fueling("А030ВС77"),
        )
        self.assertEqual(second_attempt.status, DecisionStatus.DENIED)

    def test_manual_mode_setting_is_persistent(self) -> None:
        self.assertTrue(self.repository.manual_approval_enabled(default=True))
        self.repository.set_manual_approval_enabled(False)
        self.assertFalse(self.repository.manual_approval_enabled(default=True))

    def test_display_timeout_uses_default_validates_and_persists(self) -> None:
        self.assertEqual(self.repository.display_timeout_seconds(), 10)
        self.repository.set_display_timeout_seconds(25)
        self.assertEqual(self.repository.display_timeout_seconds(), 25)
        with self.assertRaises(ValueError):
            self.repository.set_display_timeout_seconds(2)
        with self.assertRaises(ValueError):
            self.repository.set_display_timeout_seconds(121)

    def test_history_columns_keep_time_and_plate_mandatory(self) -> None:
        self.assertEqual(
            self.repository.history_visible_columns(),
            ("time", "camera", "plate", "decision", "fueling", "mode"),
        )
        self.repository.set_history_visible_columns(
            ("time", "plate", "decision", "fueling")
        )
        self.assertEqual(
            self.repository.history_visible_columns(),
            ("time", "plate", "decision", "fueling"),
        )
        with self.assertRaises(ValueError):
            self.repository.set_history_visible_columns(("plate",))
        with self.assertRaises(ValueError):
            self.repository.set_history_visible_columns(("time", "plate", "unknown"))

    def _event(self, status: DecisionStatus) -> RecognitionEvent:
        return RecognitionEvent(
            camera_id="camera-1",
            observed_at=self.observed_at,
            raw_text="A030BC77",
            normalized_plate="А030ВС77",
            ocr_confidence=0.9,
            detection_confidence=0.95,
            decision=status,
            reason="Тест",
            image_path=None,
        )


if __name__ == "__main__":
    unittest.main()
