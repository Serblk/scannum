from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from plate_guard.config import AppConfig, CameraConfig, ProjectConfig, RecognitionConfig
from plate_guard.gui import MainWindow
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
            finally:
                window.close()


if __name__ == "__main__":
    unittest.main()
