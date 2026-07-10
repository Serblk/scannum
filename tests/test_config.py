from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from plate_guard.config import ConfigError, load_config


VALID_CONFIG = """
[app]
database_path = "data/test.db"
captures_directory = "captures"
reports_directory = "reports"
timezone = "Europe/Moscow"
process_every_n_frames = 3
frame_queue_size = 8
camera_retry_seconds = 5.0
confirmations_required = 3
confirmation_window_seconds = 3.0
duplicate_cooldown_seconds = 30.0
minimum_ocr_confidence = 0.65
fueling_interval_hours = 8
manual_approval_enabled = true

[recognition]
detector_model = "detector"
ocr_model = "ocr"
device = "cpu"
model_cache_directory = "models/cache"

[[cameras]]
id = "camera-1"
name = "Камера 1"
source = 0
enabled = true
width = 1280
height = 720
"""


class ConfigTests(unittest.TestCase):
    def test_resolves_paths_relative_to_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "config.toml"
            config_path.write_text(VALID_CONFIG, encoding="utf-8")
            config = load_config(config_path)

            self.assertEqual(config.app.database_path, config_path.parent / "data/test.db")
            self.assertEqual(
                config.recognition.model_cache_directory,
                config_path.parent / "models/cache",
            )
            self.assertEqual(config.enabled_cameras[0].source, 0)

    def test_rejects_duplicate_camera_ids(self) -> None:
        duplicate = VALID_CONFIG + """
[[cameras]]
id = "camera-1"
name = "Камера 2"
source = 1
enabled = true
width = 1280
height = 720
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "config.toml"
            config_path.write_text(duplicate, encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(config_path)


if __name__ == "__main__":
    unittest.main()
