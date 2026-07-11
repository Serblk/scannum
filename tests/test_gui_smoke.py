from __future__ import annotations

import os
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QHeaderView

from plate_guard.admin import HistoryPreview
from plate_guard.config import AppConfig, CameraConfig, ProjectConfig, RecognitionConfig
from plate_guard.gui import HistoryClearConfirmation, MainWindow, _display_reason
from plate_guard.models import AccessDecision, DecisionStatus
from plate_guard.service import PlateGuardService
from plate_guard.storage import SQLiteRepository


class _NoopRecognizer:
    def recognize(self, frame: object) -> list[object]:
        return []


class GuiSmokeTests(unittest.TestCase):
    def test_main_window_contains_administration_button(self) -> None:
        application = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            camera = CameraConfig("camera-1", "Камера 1", 0)
            config = ProjectConfig(
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
                recognition=RecognitionConfig(
                    "detector", "ocr", "cpu", root / "models"
                ),
                cameras=(camera,),
            )
            repository = SQLiteRepository(config.app.database_path)
            repository.initialize()
            repository.upsert_cameras([camera])
            service = PlateGuardService(
                config,
                repository,
                _NoopRecognizer(),  # type: ignore[arg-type]
            )
            window = MainWindow(config, repository, service)
            try:
                self.assertEqual(window.admin_button.text(), "Администрирование")
                self.assertTrue(window.admin_button.isEnabled())
                self.assertEqual(window.camera_settings_button.text(), "Настройка камер")
                self.assertEqual(service.active_cameras, ())
                header = window.history.horizontalHeader()
                self.assertEqual(
                    header.sectionResizeMode(0),
                    QHeaderView.ResizeMode.ResizeToContents,
                )
                self.assertEqual(
                    header.sectionResizeMode(5),
                    QHeaderView.ResizeMode.Stretch,
                )
                cameras = tuple(
                    CameraConfig(f"camera-{index}", f"Камера {index}", index)
                    for index in range(4)
                )
                window._rebuild_video_grid(cameras)
                self.assertEqual(set(window._video_panels), {camera.id for camera in cameras})
                window._displayed_plate = "А123ВС77"
                window.plate_label.setText("А123ВС77")
                window._clear_current_display()
                self.assertEqual(window.plate_label.text(), "—")
                window._pending_event_id = 42
                window.plate_label.setText("А123ВС77")
                window._clear_current_display()
                self.assertEqual(window.plate_label.text(), "А123ВС77")
            finally:
                window.close()

    def test_denied_reason_contains_next_allowed_local_time(self) -> None:
        decision = AccessDecision(
            DecisionStatus.DENIED,
            "Ещё не прошло восемь часов",
            next_allowed_at=datetime(2026, 7, 11, 8, 43, 6, tzinfo=UTC),
        )
        text = _display_reason(
            decision.reason,
            decision,
            ZoneInfo("Europe/Moscow"),
        )
        self.assertIn("Следующий допустимый момент", text)
        self.assertIn("11.07.2026 11:43:06", text)

    def test_history_clear_confirmation_uses_russian_buttons(self) -> None:
        application = QApplication.instance() or QApplication([])
        dialog = HistoryClearConfirmation(HistoryPreview(1, 2, 3, 4, 5))
        try:
            self.assertEqual(dialog.delete_button.text(), "Да, удалить")
            self.assertEqual(dialog.cancel_button.text(), "Отмена")
            dialog.delete_button.click()
            self.assertTrue(dialog.confirmed)
        finally:
            dialog.close()


if __name__ == "__main__":
    unittest.main()
