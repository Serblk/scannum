from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from plate_guard.camera import FramePacket
from plate_guard.config import AppConfig, CameraConfig, ProjectConfig, RecognitionConfig
from plate_guard.models import DecisionStatus, PlateCandidate
from plate_guard.service import PlateGuardService
from plate_guard.storage import SQLiteRepository


class _FakeRecognizer:
    def recognize(self, frame: object) -> list[PlateCandidate]:
        return [
            PlateCandidate(
                raw_text="A030BC77",
                canonical_text="А030ВС77",
                normalized_plate="А030ВС77",
                ocr_confidence=0.95,
                detection_confidence=0.96,
                bounding_box=(5, 5, 80, 30),
            )
        ]


class _SequenceRecognizer:
    def __init__(self, plates: list[str]) -> None:
        self._plates = iter(plates)

    def recognize(self, frame: object) -> list[PlateCandidate]:
        plate = next(self._plates)
        return [
            PlateCandidate(
                raw_text=plate,
                canonical_text=plate,
                normalized_plate=plate,
                ocr_confidence=0.95,
                detection_confidence=0.96,
                bounding_box=(5, 5, 80, 30),
            )
        ]


class ServiceWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.camera = CameraConfig("camera-1", "Камера 1", 0)
        self.config = ProjectConfig(
            app=AppConfig(
                database_path=root / "data/system.db",
                captures_directory=root / "captures",
                reports_directory=root / "reports",
                timezone="Europe/Moscow",
                process_every_n_frames=1,
                frame_queue_size=2,
                camera_retry_seconds=1.0,
                confirmations_required=1,
                confirmation_window_seconds=2.0,
                duplicate_cooldown_seconds=30.0,
                minimum_ocr_confidence=0.5,
                fueling_interval_hours=8,
                manual_approval_enabled=True,
            ),
            recognition=RecognitionConfig("detector", "ocr", "cpu", root / "models"),
            cameras=(self.camera,),
        )
        self.repository = SQLiteRepository(self.config.app.database_path)
        self.repository.initialize()
        self.repository.upsert_cameras([self.camera])
        self.frame = np.zeros((60, 120, 3), dtype=np.uint8)
        self.started_at = datetime(2026, 7, 10, 9, tzinfo=UTC)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_automatic_mode_immediately_starts_timer(self) -> None:
        self.repository.set_manual_approval_enabled(False)
        events: list[tuple[int, DecisionStatus, bool]] = []
        service = self._service(events)

        self._scan(service, self.started_at)
        self._scan(service, self.started_at + timedelta(minutes=1))

        rows = list(reversed(self.repository.recent_recognitions(10)))
        self.assertEqual([row["decision"] for row in rows], ["ALLOWED", "DENIED"])
        self.assertEqual(rows[0]["outcome"], "FUELED")
        self.assertEqual(rows[0]["mode"], "AUTO")
        self.assertEqual(events[0][1], DecisionStatus.ALLOWED)
        self.assertFalse(events[0][2])

    def test_manual_mode_starts_timer_only_after_fueled_button(self) -> None:
        events: list[tuple[int, DecisionStatus, bool]] = []
        service = self._service(events)

        self._scan(service, self.started_at)
        first_id = events[-1][0]
        self.assertTrue(events[-1][2])
        self.assertIsNone(self.repository.last_confirmed_fueling("А030ВС77"))

        service.resolve_manual_decision(
            first_id,
            fueled=False,
            decided_at=self.started_at + timedelta(seconds=10),
        )
        self.assertIsNone(self.repository.last_confirmed_fueling("А030ВС77"))

        self._scan(service, self.started_at + timedelta(minutes=1))
        second_id = events[-1][0]
        self.assertNotEqual(first_id, second_id)
        service.resolve_manual_decision(
            second_id,
            fueled=True,
            decided_at=self.started_at + timedelta(minutes=1, seconds=10),
        )
        self.assertIsNotNone(self.repository.last_confirmed_fueling("А030ВС77"))

        self._scan(service, self.started_at + timedelta(minutes=2))
        self.assertEqual(events[-1][1], DecisionStatus.DENIED)

    def test_can_replace_active_camera_set_before_start(self) -> None:
        service = self._service([])
        second = CameraConfig("camera-2", "Камера 2", 1)

        service.configure_cameras((self.camera, second))

        self.assertEqual(service.active_cameras, (self.camera, second))
        service.configure_cameras(())
        self.assertEqual(service.active_cameras, ())

    def test_ambiguous_ocr_creates_one_review_without_fueling(self) -> None:
        recognizer = _SequenceRecognizer(
            ["С888КТ197", "С888КГ197", "С888КТ197", "С888КГ197"]
        )
        service = PlateGuardService(
            self.config,
            self.repository,
            recognizer,  # type: ignore[arg-type]
        )

        self._scan(service, self.started_at)

        rows = self.repository.recent_recognitions(10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["decision"], "REVIEW")
        self.assertIsNone(rows[0]["normalized_plate"])
        self.assertIn("OCR не смог однозначно определить номер", rows[0]["reason"])
        self.assertIsNone(self.repository.last_confirmed_fueling("С888КТ197"))

    def _service(
        self, events: list[tuple[int, DecisionStatus, bool]]
    ) -> PlateGuardService:
        return PlateGuardService(
            self.config,
            self.repository,
            _FakeRecognizer(),  # type: ignore[arg-type]
            event_handler=lambda event_id, event, decision, pending: events.append(
                (event_id, decision.status, pending)
            ),
        )

    def _packet(self, observed_at: datetime) -> FramePacket:
        return FramePacket("camera-1", observed_at, self.frame)

    def _scan(self, service: PlateGuardService, started_at: datetime) -> None:
        for seconds in range(4):
            service._process_frame(self._packet(started_at + timedelta(seconds=seconds)))


if __name__ == "__main__":
    unittest.main()
