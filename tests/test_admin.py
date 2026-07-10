from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from plate_guard.admin import (
    AdminLockedError,
    AdminService,
    AuthenticationError,
    PasswordPolicyError,
)
from plate_guard.config import CameraConfig
from plate_guard.models import DecisionStatus, RecognitionEvent
from plate_guard.storage import SQLiteRepository


class AdminServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.captures = self.root / "captures"
        self.repository = SQLiteRepository(self.root / "data/system.db")
        self.repository.initialize()
        self.repository.upsert_cameras(
            [CameraConfig("camera-1", "Камера 1", 0)]
        )
        self.now = [100.0]
        self.admin = AdminService(
            self.repository,
            self.captures,
            clock=lambda: self.now[0],
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_first_password_is_hashed_and_verifiable(self) -> None:
        self.admin.create_password("надёжный-пароль", "надёжный-пароль")
        stored = self.repository.admin_password_hash()
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertNotIn("надёжный-пароль", stored)
        self.assertTrue(stored.startswith("$argon2id$"))
        self.admin.verify_password("надёжный-пароль")

    def test_password_policy_rejects_short_and_mismatched_values(self) -> None:
        with self.assertRaises(PasswordPolicyError):
            self.admin.create_password("короткий", "короткий")
        with self.assertRaises(PasswordPolicyError):
            self.admin.create_password("достаточно-длинный", "другой-пароль")

    def test_five_failures_lock_login_for_thirty_seconds(self) -> None:
        self.admin.create_password("правильный-пароль", "правильный-пароль")
        for _ in range(5):
            with self.assertRaises(AuthenticationError):
                self.admin.verify_password("неверный-пароль")
        with self.assertRaises(AdminLockedError):
            self.admin.verify_password("правильный-пароль")

        self.now[0] += 31
        self.admin.verify_password("правильный-пароль")

    def test_clear_removes_history_and_photos_but_keeps_password_and_settings(self) -> None:
        self.admin.create_password("надёжный-пароль", "надёжный-пароль")
        self.repository.set_manual_approval_enabled(False)
        self.repository.record_recognition(
            RecognitionEvent(
                camera_id="camera-1",
                observed_at=datetime(2026, 7, 10, 9, tzinfo=UTC),
                raw_text="A030BC77",
                normalized_plate="А030ВС77",
                ocr_confidence=0.9,
                detection_confidence=0.95,
                decision=DecisionStatus.ALLOWED,
                reason="Тест",
                image_path=str(self.captures / "camera-1/test.jpg"),
            ),
            auto_confirm=True,
        )
        photo = self.captures / "camera-1/test.jpg"
        photo.parent.mkdir(parents=True)
        photo.write_bytes(b"test-image")
        (self.captures / ".gitkeep").write_text("", encoding="utf-8")

        result = self.admin.clear_history_and_photos("надёжный-пароль")

        self.assertEqual(result.preview.recognitions, 1)
        self.assertEqual(result.preview.photos, 1)
        self.assertEqual(self.repository.history_summary()["recognitions"], 0)
        self.assertFalse(photo.exists())
        self.assertTrue((self.captures / ".gitkeep").exists())
        self.assertTrue(self.admin.password_is_configured)
        self.assertFalse(self.repository.manual_approval_enabled(default=True))

    def test_wrong_password_does_not_clear_anything(self) -> None:
        self.admin.create_password("надёжный-пароль", "надёжный-пароль")
        photo = self.captures / "camera-1/test.jpg"
        photo.parent.mkdir(parents=True)
        photo.write_bytes(b"test-image")

        with self.assertRaises(AuthenticationError):
            self.admin.clear_history_and_photos("неверный-пароль")

        self.assertTrue(photo.exists())


if __name__ == "__main__":
    unittest.main()
